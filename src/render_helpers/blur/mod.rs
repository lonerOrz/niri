// Ported from https://github.com/nferhat/fht-compositor/blob/main/src/renderer/blur/mod.rs

pub mod element;
pub mod optimized_blur_texture_element;
pub(super) mod shader;

use std::cell::{RefCell, RefMut};
use std::rc::Rc;
use std::sync::MutexGuard;
use std::time::{Duration, Instant};

/// 简单的 FBO RAII：在 drop 时删除 fbo
struct ScopedFbo {
    id: u32,
    gl: *const ffi::Gles2,
}
impl ScopedFbo {
    fn new(gl: &ffi::Gles2) -> Result<Self, GlesError> {
        let mut id = 0;
        unsafe { gl.GenFramebuffers(1, &mut id as *mut _) };
        if id == 0 {
            return Err(GlesError::FramebufferBindingError);
        }
        Ok(Self { id, gl: gl as *const _ })
    }
}
impl std::fmt::Debug for ScopedFbo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScopedFbo {{ id: {} }}", self.id)
    }
}
impl Drop for ScopedFbo {
    fn drop(&mut self) {
        unsafe {
            // Safety: caller ensures gl pointer is valid while ScopedFbo alive
            if self.gl.is_null() { return; } // Added null check
            (*self.gl).DeleteFramebuffers(1, &mut (self.id) as *mut _);
        }
    }
}

/// 在创建时记住 prev binding，drop 时恢复
struct FboBindingGuard {
    pub prev: i32,
    gl: *const ffi::Gles2,
}
impl FboBindingGuard {
    fn new(gl: &ffi::Gles2) -> Self {
        let mut prev = 0;
        unsafe { gl.GetIntegerv(ffi::FRAMEBUFFER_BINDING, &mut prev as *mut _) };
        Self { prev, gl }
    }
}
impl Drop for FboBindingGuard {
    fn drop(&mut self) {
        unsafe {
            (*self.gl).BindFramebuffer(ffi::FRAMEBUFFER, self.prev as u32);
        }
    }
}


use anyhow::Context;
use glam::{Mat3, Vec2};
use niri_config::Blur;
use shader::BlurShaders;
use smithay::backend::renderer::element::surface::WaylandSurfaceRenderElement;
use smithay::backend::renderer::element::AsRenderElements;
use smithay::backend::renderer::gles::format::fourcc_to_gl_formats;
use smithay::backend::renderer::gles::{ffi, Capability, GlesError, GlesRenderer, GlesTexture};
use smithay::backend::renderer::sync::SyncPoint;
use smithay::backend::renderer::{Bind, Blit, Frame, Offscreen, Renderer, Texture, TextureFilter};
use smithay::desktop::LayerMap;
use smithay::output::Output;
use smithay::reexports::gbm::Format;
use smithay::utils::{Buffer, Physical, Point, Rectangle, Scale, Size, Transform};
use smithay::wayland::shell::wlr_layer::Layer;

use super::render_data::RendererData;
use super::render_elements;
use super::shaders::Shaders;
use crate::render_helpers::renderer::NiriRenderer;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum CurrentBuffer {
    /// We are currently sampling from normal buffer, and rendering in the swapped/alternative.
    #[default]
    Normal,
    /// We are currently sampling from swapped buffer, and rendering in the normal.
    Swapped,
}

impl CurrentBuffer {
    pub fn swap(&mut self) {
        *self = match self {
            // sampled from normal, render to swapped
            Self::Normal => Self::Swapped,
            // sampled fro swapped, render to normal next
            Self::Swapped => Self::Normal,
        }
    }
}

/// Effect framebuffers associated with each output.
pub struct EffectsFramebuffers {
    /// Contains the main buffer blurred contents
    pub optimized_blur: GlesTexture,
    /// Whether the optimizer blur buffer is dirty
    pub optimized_blur_rerender_at: Option<Instant>,
    // /// Contains the original pixels before blurring to draw with in case of artifacts.
    // blur_saved_pixels: GlesTexture,
    // The blur algorithms (dual-kawase) swaps between these two whenever scaling the image
    effects: GlesTexture,
    effects_swapped: GlesTexture,
    /// The buffer we are currently rendering/sampling from.
    ///
    /// In order todo the up/downscaling, we render into different buffers. On each pass, we render
    /// into a different buffer with downscaling/upscaling (depending on which pass we are at)
    ///
    /// One exception is that if we are on the first pass, we are on [`CurrentBuffer::Initial`], we
    /// are sampling from [`Self::blit_buffer`] from initial screen contents.
    current_buffer: CurrentBuffer,
    pub last_blur_update_time: Instant,
    pub last_set_dirty_call_time: Instant,
    pub pending_blur_due_to_throttle: bool,
}

type EffectsFramebufffersUserData = Rc<RefCell<EffectsFramebuffers>>;

fn get_rerender_at() -> Option<Instant> {
    Some(Instant::now())
}

impl EffectsFramebuffers {
    /// Get the assiciated [`EffectsFramebuffers`] with this output.
    pub fn get<'a>(output: &'a Output) -> RefMut<'a, Self> {
        let user_data = output
            .user_data()
            .get::<EffectsFramebufffersUserData>()
            .unwrap();
        RefCell::borrow_mut(user_data)
    }
    pub fn set_dirty(output: &Output) {
        const THROTTLE_INTERVAL_MS: u64 = 33;

        let Some(mut fx_buffers) = output
            .user_data()
            .get::<EffectsFramebufffersUserData>()
            .map(|user_data| RefCell::borrow_mut(user_data))
        else {
            return;
        };
        let now = Instant::now();

        // Determine if we are in a "rapid change" phase
        let is_rapid_change = now.duration_since(fx_buffers.last_set_dirty_call_time)
            < Duration::from_millis(THROTTLE_INTERVAL_MS);
        fx_buffers.last_set_dirty_call_time = now; // Update for the next call

        // Check if we should throttle
        let should_throttle = is_rapid_change
            && now.duration_since(fx_buffers.last_blur_update_time)
                < Duration::from_millis(THROTTLE_INTERVAL_MS);

        if should_throttle {
            fx_buffers.pending_blur_due_to_throttle = true;
            return;
        }

        // Always mark for immediate re-render if not throttled
        fx_buffers.optimized_blur_rerender_at = Some(Instant::now());
        fx_buffers.last_blur_update_time = now;

        if fx_buffers.pending_blur_due_to_throttle {
            fx_buffers.pending_blur_due_to_throttle = false;
            fx_buffers.optimized_blur_rerender_at = Some(Instant::now());
            fx_buffers.last_blur_update_time = now;
        }
    }

    /// Initialize the [`EffectsFramebuffers`] for an [`Output`].
    ///
    /// The framebuffers handles live inside the Output's user data, use [`Self::get`] to access
    /// them.
    pub fn init_for_output(output: Output, renderer: &mut impl NiriRenderer) {
        let renderer = renderer.as_gles_renderer();
        let output_size = output.current_mode().unwrap().size;

        fn create_buffer(
            renderer: &mut GlesRenderer,
            size: Size<i32, Physical>,
        ) -> Result<GlesTexture, GlesError> {
            renderer.create_buffer(
                Format::Abgr8888,
                size.to_logical(1).to_buffer(1, Transform::Normal),
            )
        }

        let this = EffectsFramebuffers {
            optimized_blur: create_buffer(renderer, output_size).unwrap(),
            optimized_blur_rerender_at: get_rerender_at(),
            effects: create_buffer(renderer, output_size).unwrap(),
            effects_swapped: create_buffer(renderer, output_size).unwrap(),
            current_buffer: CurrentBuffer::Normal,
            last_blur_update_time: Instant::now(),
            last_set_dirty_call_time: Instant::now(),
            pending_blur_due_to_throttle: false,
        };

        let user_data = output.user_data();
        assert!(
            user_data.insert_if_missing(|| Rc::new(RefCell::new(this))),
            "EffectsFrambuffers::init_for_output should only be called once!"
        );
    }

    /// Update the [`EffectsFramebuffers`] for an [`Output`].
    ///
    /// You should call this if the output's scale/size changes
    pub fn update_for_output(
        output: Output,
        renderer: &mut impl NiriRenderer,
    ) -> Result<(), GlesError> {
        let renderer = renderer.as_gles_renderer();
        let mut fx_buffers = Self::get(&output);
        // Cache commonly accessed values to reduce repeated calls
        let output_mode = output.current_mode();
        if output_mode.is_none() {
            return Ok(());
        }
        let output_size = output_mode.unwrap().size;

        fn create_buffer(
            renderer: &mut GlesRenderer,
            size: Size<i32, Physical>,
        ) -> Result<GlesTexture, GlesError> {
            renderer.create_buffer(
                Format::Abgr8888,
                size.to_logical(1).to_buffer(1, Transform::Normal),
            )
        }

        *fx_buffers = EffectsFramebuffers {
            optimized_blur: create_buffer(renderer, output_size)?,
            optimized_blur_rerender_at: get_rerender_at(),
            effects: create_buffer(renderer, output_size)?,
            effects_swapped: create_buffer(renderer, output_size)?,
            current_buffer: CurrentBuffer::Normal,
            last_blur_update_time: Instant::now(),
            last_set_dirty_call_time: Instant::now(),
            pending_blur_due_to_throttle: false,
        };

        Ok(())
    }

    /// Render the optimized blur buffer again
    pub fn update_optimized_blur_buffer(
        &mut self,
        renderer: &mut GlesRenderer,
        layer_map: MutexGuard<'_, LayerMap>,
        output: &Output,
        scale: Scale<f64>,
        config: Blur,
    ) -> anyhow::Result<()> {
        // Cache the rerender time to avoid multiple Option unwraps
        let rerender_time = self.optimized_blur_rerender_at;
        if rerender_time.is_none() {
            return Ok(());
        }

        // Early exit if not yet time to rerender
        if rerender_time.unwrap() > Instant::now() {
            return Ok(());
        }

        self.optimized_blur_rerender_at = None;

        // Check for valid output mode before proceeding
        let output_mode = output.current_mode();
        if output_mode.is_none() {
            return Ok(());
        }
        let output_size = output_mode.unwrap().size;

        // Early exit for zero-area operations
        if output_size.w <= 0 || output_size.h <= 0 {
            return Ok(());
        }

        // Early exit for disabled blur
        if !config.on {
            return Ok(());
        }

        // Early exit for zero-radius blur
        if config.radius.0 <= 0.0 {
            return Ok(());
        }

        // first render layer shell elements
        // NOTE: We use Blur::DISABLED since we should not include blur with Background/Bottom
        // layer shells

        let mut elements = Vec::with_capacity(16); // Pre-allocate reasonable capacity to reduce reallocations

        // Use chain iterator instead of collecting into intermediate vectors for better performance
        for layer in layer_map
            .layers_on(Layer::Background)
            .chain(layer_map.layers_on(Layer::Bottom))
            .rev()
        {
            let layer_geo = layer_map.layer_geometry(layer).unwrap();
            let location = layer_geo.loc.to_physical_precise_round(scale);
            // Directly extend without intermediate collection for better performance
            elements.extend(
                layer.render_elements::<WaylandSurfaceRenderElement<_>>(
                    renderer, location, scale, 1.0,
                ),
            );
        }

        let mut fb = renderer.bind(&mut self.effects).unwrap();
        // Use the already validated output_size
        let output_size = output_mode.unwrap().size;

        let _ = render_elements(
            renderer,
            &mut fb,
            output_size,
            scale,
            Transform::Normal,
            elements.iter(),
        )
        .expect("failed to render for optimized blur buffer");
        drop(fb);

        self.current_buffer = CurrentBuffer::Normal;

        let shaders = Shaders::get(renderer).blur.clone();

        // Cache output dimensions to avoid repeated unwrap calls
        let output_size = output.current_mode().unwrap().size;

        // NOTE: If we only do one pass its kinda ugly, there must be at least
        // n=2 passes in order to have good sampling
        let half_pixel = [
            0.5 / (output_size.w as f32 / 2.0),
            0.5 / (output_size.h as f32 / 2.0),
        ];

        // Adaptive passes calculation for better performance vs quality trade-off
        let (downsample_passes, upsample_passes) = {
            let base_passes = config.passes.max(1); // Ensure at least 1 pass
            let radius = config.radius.0;

            // Adjust passes based on blur radius for optimal performance
            let adjusted_passes = if radius < 1.0 {
                // Very small blur - reduce passes significantly
                1
            } else if radius < 3.0 {
                // Small blur - moderate passes
                base_passes.min(2)
            } else if radius < 6.0 {
                // Medium blur - standard passes
                base_passes
            } else {
                // Large blur - increase passes for better quality
                base_passes.min(6) // Cap to prevent excessive computation
            };

            // Separate downsample and upsample passes for better control
            let down = adjusted_passes;
            let up = adjusted_passes;

            (down, up)
        };

        // Reduce computation for minimal blur effect
        // Use a small epsilon to handle floating point precision issues
        if config.radius.0 >= 0.1f64 {
            let mut sync_point: Option<SyncPoint> = None;

            // Downsample passes
            for _ in 0..downsample_passes {
                if let Some(sync) = sync_point {
                    sync.wait().unwrap();
                }
                let (sample_buffer, render_buffer) = self.buffers();
                sync_point = Some(render_blur_pass_with_frame(
                    renderer,
                    sample_buffer,
                    render_buffer,
                    &shaders.down,
                    half_pixel,
                    config,
                )?);
                self.current_buffer.swap();
            }

            let half_pixel = [
                0.5 / (output_size.w as f32 * 2.0),
                0.5 / (output_size.h as f32 * 2.0),
            ];
            // FIXME: Why we need inclusive here but down is exclusive?
            // Upsample passes
            for _ in 0..upsample_passes {
                if let Some(sync) = sync_point {
                    sync.wait().unwrap();
                }
                let (sample_buffer, render_buffer) = self.buffers();
                sync_point = Some(render_blur_pass_with_frame(
                    renderer,
                    sample_buffer,
                    render_buffer,
                    &shaders.up,
                    half_pixel,
                    config,
                )?);
                self.current_buffer.swap();
            }
        }
        // For very small radius, skip blur computation entirely

        // Now blit from the last render buffer into optimized_blur
        // We are already bound so its just a blit
        let tex_fb = renderer.bind(&mut self.effects).unwrap();
        let mut optimized_blur_fb = renderer.bind(&mut self.optimized_blur).unwrap();

        let _ = renderer.blit(
            &tex_fb,
            &mut optimized_blur_fb,
            Rectangle::from_size(output_size),
            Rectangle::from_size(output_size),
            TextureFilter::Linear,
        )?;

        Ok(())
    }

    /// Get the sample and render buffers.
    pub fn buffers(&mut self) -> (&GlesTexture, &mut GlesTexture) {
        match self.current_buffer {
            CurrentBuffer::Normal => (&self.effects, &mut self.effects_swapped),
            CurrentBuffer::Swapped => (&self.effects_swapped, &mut self.effects),
        }
    }
}

pub(super) unsafe fn get_main_buffer_blur(
    gl: &ffi::Gles2,
    fx_buffers: &mut EffectsFramebuffers,
    shaders: &BlurShaders,
    blur_config: Blur,
    projection_matrix: Mat3,
    scale: i32,
    vbos: &[u32; 2],
    debug: bool,
    supports_instancing: bool,
    // dst is the region that we want blur on
    dst: Rectangle<i32, Physical>,
    is_tty: bool,
    is_shared: bool,
) -> Result<GlesTexture, GlesError> {
    let tex_size = fx_buffers
        .effects
        .size()
        .to_logical(1, Transform::Normal)
        .to_physical(scale);

    let dst_expanded = {
        let mut dst = dst;
        let size =
            (2f32.powi(blur_config.passes as i32 + 1) * blur_config.radius.0 as f32).ceil() as i32;
        dst.loc -= Point::from((size, size));
        dst.size += Size::from((size, size)).upscale(2);
        // Clamp the expanded destination to the texture size to prevent out-of-bounds blitting.
        Rectangle::new(
            Point::from((dst.loc.x.max(0), dst.loc.y.max(0))),
            Size::from((
                dst.size.w.min(tex_size.w - dst.loc.x.max(0)),
                dst.size.h.min(tex_size.h - dst.loc.y.max(0)),
            )),
        )
    };

    let binding_guard = FboBindingGuard::new(gl);

    let (sample_buffer, _) = fx_buffers.buffers();

    // Check GLES version for BlitFramebuffer support
    let gles_version_str = gl.GetString(ffi::VERSION);
    let gles_version = std::ffi::CStr::from_ptr(gles_version_str as *const i8)
        .to_str()
        .unwrap_or("Unknown GLES Version");

    let mut major_version = 2;
    let mut minor_version = 0;
    if let Some(version_part) =
        gles_version.get(gles_version.find("OpenGL ES ").unwrap_or(0) + "OpenGL ES ".len()..)
    {
        let mut parts = version_part.split('.');
        major_version = parts.next().and_then(|s| s.parse().ok()).unwrap_or(2);
        minor_version = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    }

    if major_version < 3 {
        error!(
            "TrueBlur requires GLES 3.0 for blitting, but found GLES version: {}",
            gles_version
        );
        return Err(GlesError::BlitError);
    }

    let _render_buffer_fbo_guard = ScopedFbo::new(gl)?;
    let render_buffer_fbo = _render_buffer_fbo_guard.id;

    let supports_memory_barrier = major_version > 3 || (major_version == 3 && minor_version >= 1);

    // First get a fbo for the texture we are about to read into
    let _sample_fbo_guard = ScopedFbo::new(gl)?;
    let sample_fbo = _sample_fbo_guard.id;
    {
        gl.BindFramebuffer(ffi::FRAMEBUFFER, sample_fbo);
        gl.FramebufferTexture2D(
            ffi::FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            sample_buffer.tex_id(),
            0,
        );
        gl.Clear(ffi::COLOR_BUFFER_BIT);
        let status = gl.CheckFramebufferStatus(ffi::FRAMEBUFFER);
        if status != ffi::FRAMEBUFFER_COMPLETE {
            // gl.DeleteFramebuffers(1, &mut sample_fbo as *mut _); // Handled by ScopedFbo
            // gl.BindFramebuffer(ffi::FRAMEBUFFER, prev_fbo as u32); // Handled by FboBindingGuard
            return Err(GlesError::FramebufferBindingError);
        }
    }

    {
        // NOTE: We are assured that the size of the effects texture is the same
        // as the bound fbo size, so blitting uses dst immediatly
        gl.BindFramebuffer(ffi::READ_FRAMEBUFFER, binding_guard.prev as u32);
        gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, sample_fbo);

        if is_tty {
            let src_x0 = dst_expanded.loc.x;
            let src_y0 = dst_expanded.loc.y;
            let src_x1 = dst_expanded.loc.x + dst_expanded.size.w;
            let src_y1 = dst_expanded.loc.y + dst_expanded.size.h;
            let dst_x0 = src_x0;
            let dst_y0 = src_y0;
            let dst_x1 = src_x1;
            let dst_y1 = src_y1;

            gl.BlitFramebuffer(
                src_x0,
                src_y0,
                src_x1,
                src_y1,
                dst_x0,
                dst_y0,
                dst_x1,
                dst_y1,
                ffi::COLOR_BUFFER_BIT,
                ffi::LINEAR,
            );
        } else {
            let src_x0 = dst_expanded.loc.x;
            let src_x1 = dst_expanded.loc.x + dst_expanded.size.w;
            let src_y0 = dst_expanded.loc.y;
            let src_y1 = dst_expanded.loc.y + dst_expanded.size.h;
            let dst_y0 = dst_expanded.loc.y;
            let dst_y1 = dst_expanded.loc.y + dst_expanded.size.h;

            gl.BlitFramebuffer(
                src_x0,
                src_y0,
                src_x1,
                src_y1,
                src_x0,
                dst_y0,
                src_x1,
                dst_y1,
                ffi::COLOR_BUFFER_BIT,
                ffi::LINEAR,
            );
        }

        let error = gl.GetError();
        if error != ffi::NO_ERROR {
            error!("gl.BlitFramebuffer failed with error: {:#x}", error);
            // gl.BindFramebuffer(ffi::FRAMEBUFFER, prev_fbo as u32); // Handled by FboBindingGuard
            return Err(GlesError::BlitError);
        }
        // gl.BindFramebuffer(ffi::FRAMEBUFFER, prev_fbo as u32); // Handled by FboBindingGuard

        // Add a barrier here to prevent a read-after-write hazard.
        // The subsequent blur passes will read from the texture we just blitted to.
        if is_shared {
            gl.Finish();
        } else if supports_memory_barrier {
            gl.MemoryBarrier(ffi::TEXTURE_FETCH_BARRIER_BIT);
        } else {
            // Fallback for GLES < 3.1
            gl.Finish();
        }

        {
            let passes = blur_config.passes;
            let half_pixel = [
                0.5 / (tex_size.w as f32 / 2.0),
                0.5 / (tex_size.h as f32 / 2.0),
            ];
            for i in 0..passes {
                let (sample_buffer, render_buffer) = fx_buffers.buffers();
                let damage = dst_expanded.downscale(1 << (i + 1));
                render_blur_pass_with_gl(
                    gl,
                    vbos,
                    debug,
                    supports_instancing,
                    projection_matrix,
                    sample_buffer,
                    render_buffer,
                    scale,
                    &shaders.down,
                    half_pixel,
                    blur_config.clone(),
                    damage,
                    is_shared,
                    supports_memory_barrier,
                    render_buffer_fbo,
                )?;
                fx_buffers.current_buffer.swap();
            }

            let half_pixel = [
                0.5 / (tex_size.w as f32 * 2.0),
                0.5 / (tex_size.h as f32 * 2.0),
            ];
            for i in 0..passes {
                let (sample_buffer, render_buffer) = fx_buffers.buffers();
                let damage = dst_expanded.downscale(1 << (passes - 1 - i));
                render_blur_pass_with_gl(
                    gl,
                    &vbos,
                    debug,
                    supports_instancing,
                    projection_matrix,
                    sample_buffer,
                    render_buffer,
                    scale,
                    &shaders.up,
                    half_pixel,
                    blur_config.clone(),
                    damage,
                    is_shared,
                    supports_memory_barrier,
                    render_buffer_fbo,
                )?;
                fx_buffers.current_buffer.swap();
            }
        }

            // Cleanup (Handled by RAII guards)
            {
                // gl.DeleteFramebuffers(1, &mut sample_fbo as *mut _); // Handled by ScopedFbo
                // gl.DeleteFramebuffers(1, &mut render_buffer_fbo as *mut _); // Handled by ScopedFbo
                // gl.BindFramebuffer(ffi::FRAMEBUFFER, prev_fbo as u32); // Handled by FboBindingGuard
            }
        Ok(fx_buffers.effects.clone())
    }
}

// Renders a blur pass using a GlesFrame with syncing and fencing provided by smithay. Used for
// updating optimized blur buffer since we are not yet rendering.
fn render_blur_pass_with_frame(
    renderer: &mut GlesRenderer,
    sample_buffer: &GlesTexture,
    render_buffer: &mut GlesTexture,
    blur_program: &shader::BlurShader,
    half_pixel: [f32; 2],
    config: Blur,
) -> anyhow::Result<SyncPoint> {
    // We use a texture render element with a custom GlesTexProgram in order todo the blurring
    // At least this is what swayfx/scenefx do, but they just use gl calls directly.
    let size = sample_buffer.size().to_logical(1, Transform::Normal);

    let vbos = RendererData::get(renderer).vbos;

    let mut fb = renderer.bind(render_buffer)?;
    // Using GlesFrame since I want to use a custom program
    let mut frame = renderer
        .render(&mut fb, size.to_physical(1), Transform::Normal)
        .context("failed to create frame")?;

    let supports_instaning = frame.capabilities().contains(&Capability::Instancing);
    let debug = !frame.debug_flags().is_empty();
    let projection = Mat3::from_cols_array(frame.projection());

    let tex_size = sample_buffer.size();
    let src = Rectangle::from_size(sample_buffer.size()).to_f64();
    let dst = Rectangle::from_size(size).to_physical(1);

    frame.with_context(|gl| unsafe {
        // We are doing basically what Frame::render_texture_from_to does, but our own shader struct
        // instead. This allows me to get into the gl plumbing.

        let mut mat = Mat3::IDENTITY;
        let src_size = sample_buffer.size().to_f64();

        if tex_size.is_empty() || src_size.is_empty() {
            return Ok(());
        }

        let mut tex_mat = build_texture_mat(src, dst, tex_size, Transform::Normal);
        if sample_buffer.is_y_inverted() {
            tex_mat *= Mat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]);
        }

        // Since we are just rendering onto the offsreen buffer, the vertices to draw are only 4
        let damage_rect = [
            dst.loc.x as f32,
            dst.loc.y as f32,
            dst.size.w as f32,
            dst.size.h as f32,
        ];

        let mut vertices = Vec::with_capacity(4);
        let damage_len = if supports_instaning {
            vertices.extend(damage_rect);
            vertices.len() / 4
        } else {
            for _ in 0..6 {
                // Add the 4 f32s per damage rectangle for each of the 6 vertices.
                vertices.extend_from_slice(&damage_rect);
            }

            1
        };

        mat *= projection;

        // SAFETY: internal texture should always have a format
        // We also use Abgr8888 which is known and confirmed
        let (internal_format, _, _) =
            fourcc_to_gl_formats(sample_buffer.format().unwrap()).unwrap();
        let variant = blur_program.variant_for_format(Some(internal_format), false);

        let program = if debug {
            &variant.debug
        } else {
            &variant.normal
        };

        // NOTE: We know that this texture is always opaque so skip on some logic checks and
        // directly render. The following code is from GlesRenderer::render_texture
        gl.Disable(ffi::BLEND);

        gl.ActiveTexture(ffi::TEXTURE0);
        gl.BindTexture(ffi::TEXTURE_2D, sample_buffer.tex_id());
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::CLAMP_TO_EDGE as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::CLAMP_TO_EDGE as i32);
        gl.UseProgram(program.program);

        gl.Uniform1i(program.uniform_tex, 0);
        gl.UniformMatrix3fv(
            program.uniform_matrix,
            1,
            ffi::FALSE,
            mat.as_ref() as *const f32,
        );
        gl.UniformMatrix3fv(
            program.uniform_tex_matrix,
            1,
            ffi::FALSE,
            tex_mat.as_ref() as *const f32,
        );
        if program.uniform_alpha != -1 {
            gl.Uniform1f(program.uniform_alpha, 1.0);
        }
        gl.Uniform1f(program.uniform_radius, config.radius.0 as f32);
        gl.Uniform2f(program.uniform_half_pixel, half_pixel[0], half_pixel[1]);

        gl.EnableVertexAttribArray(program.attrib_vert as u32);
        gl.BindBuffer(ffi::ARRAY_BUFFER, vbos[0]);
        gl.VertexAttribPointer(
            program.attrib_vert as u32,
            2,
            ffi::FLOAT,
            ffi::FALSE,
            0,
            std::ptr::null(),
        );

        // vert_position
        gl.EnableVertexAttribArray(program.attrib_vert_position as u32);
        gl.BindBuffer(ffi::ARRAY_BUFFER, 0);

        gl.VertexAttribPointer(
            program.attrib_vert_position as u32,
            4,
            ffi::FLOAT,
            ffi::FALSE,
            0,
            vertices.as_ptr() as *const _,
        );

        if supports_instaning {
            gl.VertexAttribDivisor(program.attrib_vert as u32, 0);
            gl.VertexAttribDivisor(program.attrib_vert_position as u32, 1);
            gl.DrawArraysInstanced(ffi::TRIANGLE_STRIP, 0, 4, damage_len as i32);
        } else {
            let count = damage_len * 6;
            gl.DrawArrays(ffi::TRIANGLES, 0, count as i32);
        }

        gl.BindTexture(ffi::TEXTURE_2D, 0);
        gl.DisableVertexAttribArray(program.attrib_vert as u32);
        gl.DisableVertexAttribArray(program.attrib_vert_position as u32);

        gl.Enable(ffi::BLEND);
        gl.BlendFunc(ffi::ONE, ffi::ONE_MINUS_SRC_ALPHA);

        Result::<_, GlesError>::Ok(())
    })??;

    frame.finish().map_err(Into::into)
}

// Renders a blur pass using gl code bypassing smithay's Frame mechanisms
//
// When rendering blur in real-time (for windows, for example) there should not be a wait for
// fencing/finishing since this will be done when sending the fb to the output. Using a Frame
// forces us to do that.
unsafe fn render_blur_pass_with_gl(
    gl: &ffi::Gles2,
    vbos: &[u32; 2],
    debug: bool,
    supports_instancing: bool,
    projection_matrix: Mat3,
    // The buffers used for blurring
    sample_buffer: &GlesTexture,
    render_buffer: &mut GlesTexture,
    scale: i32,
    // The current blur program + config
    blur_program: &shader::BlurShader,
    half_pixel: [f32; 2],
    config: Blur,
    // dst is the region that should have blur
    // it gets up/downscaled with passes
    _damage: Rectangle<i32, Physical>,
    is_shared: bool,
    supports_memory_barrier: bool,
    render_buffer_fbo: u32, // New parameter
) -> Result<(), GlesError> {
    let _binding_guard = FboBindingGuard::new(gl);
    let tex_size = sample_buffer.size();
    let src = Rectangle::from_size(tex_size.to_f64());
    let dest = src
        .to_logical(1.0, Transform::Normal, &src.size)
        .to_physical(scale as f64)
        .to_i32_round();

    let damage = dest;

    // PERF: Instead of taking the whole src/dst as damage, adapt the code to run with only the
    // damaged window? This would cause us to make a custom WaylandSurfaceRenderElement to blur out
    // stuff. Complicated.

    // First bind to our render buffer
    {
        gl.BindFramebuffer(ffi::FRAMEBUFFER, render_buffer_fbo);
        gl.FramebufferTexture2D(
            ffi::FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            render_buffer.tex_id(),
            0,
        );

        let status = gl.CheckFramebufferStatus(ffi::FRAMEBUFFER);
        if status != ffi::FRAMEBUFFER_COMPLETE {
            return Err(GlesError::FramebufferBindingError);
        }
    }

    {
        let mat = projection_matrix;
        // NOTE: We are assured that tex_size != 0, and src.size != too (by damage tracker)
        let tex_mat = build_texture_mat(src, dest, tex_size, Transform::Normal);

        // FIXME: Use actual damage for this? Would require making a custom window render element
        // that includes blur and whatnot to get the damage for the window only
        let damage_rect = [
            damage.loc.x as f32,
            damage.loc.y as f32,
            damage.size.w as f32,
            damage.size.h as f32,
        ];

        let mut vertices = Vec::with_capacity(4);
        let damage_len = if supports_instancing {
            vertices.extend(damage_rect);
            vertices.len() / 4
        } else {
            for _ in 0..6 {
                // Add the 4 f32s per damage rectangle for each of the 6 vertices.
                vertices.extend_from_slice(&damage_rect);
            }

            1
        };

        // SAFETY: internal texture should always have a format
        // We also use Abgr8888 which is known and confirmed
        let (internal_format, _, _) =
            fourcc_to_gl_formats(sample_buffer.format().unwrap()).unwrap();
        let variant = blur_program.variant_for_format(Some(internal_format), false);

        let program = if debug {
            &variant.debug
        } else {
            &variant.normal
        };

        gl.Disable(ffi::BLEND);

        gl.ActiveTexture(ffi::TEXTURE0);
        gl.BindTexture(ffi::TEXTURE_2D, sample_buffer.tex_id());
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::CLAMP_TO_EDGE as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::CLAMP_TO_EDGE as i32);
        gl.UseProgram(program.program);

        gl.Uniform1i(program.uniform_tex, 0);
        gl.UniformMatrix3fv(
            program.uniform_matrix,
            1,
            ffi::FALSE,
            mat.as_ref() as *const f32,
        );
        gl.UniformMatrix3fv(
            program.uniform_tex_matrix,
            1,
            ffi::FALSE,
            tex_mat.as_ref() as *const f32,
        );
        if program.uniform_alpha != -1 {
            gl.Uniform1f(program.uniform_alpha, 1.0);
        }
        gl.Uniform1f(program.uniform_radius, config.radius.0 as f32);
        gl.Uniform2f(program.uniform_half_pixel, half_pixel[0], half_pixel[1]);

        gl.EnableVertexAttribArray(program.attrib_vert as u32);
        gl.BindBuffer(ffi::ARRAY_BUFFER, vbos[0]);
        gl.VertexAttribPointer(
            program.attrib_vert as u32,
            2,
            ffi::FLOAT,
            ffi::FALSE,
            0,
            std::ptr::null(),
        );

        // vert_position (using vbos[1] for vertex position data)
        gl.EnableVertexAttribArray(program.attrib_vert_position as u32);
        gl.BindBuffer(ffi::ARRAY_BUFFER, vbos[1]); // Assuming vbos[1] is for vert_position
        gl.BufferData(
            ffi::ARRAY_BUFFER,
            (vertices.len() * std::mem::size_of::<f32>()) as ffi::types::GLsizeiptr,
            vertices.as_ptr() as *const _,
            ffi::STREAM_DRAW,
        );

        gl.VertexAttribPointer(
            program.attrib_vert_position as u32,
            4,
            ffi::FLOAT,
            ffi::FALSE,
            0,
            std::ptr::null(), // Offset into the bound buffer
        );

        if supports_instancing {
            gl.VertexAttribDivisor(program.attrib_vert as u32, 0);
            gl.VertexAttribDivisor(program.attrib_vert_position as u32, 1);
            gl.DrawArraysInstanced(ffi::TRIANGLE_STRIP, 0, 4, damage_len as i32);
        } else {
            let count = damage_len * 6;
            gl.DrawArrays(ffi::TRIANGLES, 0, count as i32);
        }

        gl.BindTexture(ffi::TEXTURE_2D, 0);
        gl.DisableVertexAttribArray(program.attrib_vert as u32);
        gl.DisableVertexAttribArray(program.attrib_vert_position as u32);

        gl.Enable(ffi::BLEND);
        gl.BlendFunc(ffi::ONE, ffi::ONE_MINUS_SRC_ALPHA);
    }

    // After each pass, we need to add a barrier to ensure that the texture we just rendered to
    // is not read from before the rendering is complete. This is a classic read-after-write
    // hazard.
    //
    // In case of a shared context, we must use gl.Finish() to ensure that the other context
    // sees the changes. In other cases, we can use a memory barrier, which is much more
    // efficient.
    if is_shared {
        gl.Finish();
    } else if supports_memory_barrier {
        gl.MemoryBarrier(ffi::TEXTURE_FETCH_BARRIER_BIT);
    } else {
        // Fallback for GLES < 3.1
        gl.Finish();
    }

    // Clean up
    {
        gl.Enable(ffi::BLEND);
        gl.BindFramebuffer(ffi::FRAMEBUFFER, 0);
        gl.BlendFunc(ffi::ONE, ffi::ONE_MINUS_SRC_ALPHA);
    }

    Ok(())
}

// Copied from smithay, adapted to use glam structs
fn build_texture_mat(
    src: Rectangle<f64, Buffer>,
    dest: Rectangle<i32, Physical>,
    texture: Size<i32, Buffer>,
    transform: Transform,
) -> Mat3 {
    let dst_src_size = transform.transform_size(src.size);
    let scale = dst_src_size.to_f64() / dest.size.to_f64();

    let mut tex_mat = Mat3::IDENTITY;
    // first bring the damage into src scale
    tex_mat = Mat3::from_scale(Vec2::new(scale.x as f32, scale.y as f32)) * tex_mat;

    // then compensate for the texture transform
    let transform_mat = Mat3::from_cols_array(transform.matrix().as_ref());
    let translation = match transform {
        Transform::Normal => Mat3::IDENTITY,
        Transform::_90 => Mat3::from_translation(Vec2::new(0f32, dst_src_size.w as f32)),
        Transform::_180 => {
            Mat3::from_translation(Vec2::new(dst_src_size.w as f32, dst_src_size.h as f32))
        }
        Transform::_270 => Mat3::from_translation(Vec2::new(dst_src_size.h as f32, 0f32)),
        Transform::Flipped => Mat3::from_translation(Vec2::new(dst_src_size.w as f32, 0f32)),
        Transform::Flipped90 => Mat3::IDENTITY,
        Transform::Flipped180 => Mat3::from_translation(Vec2::new(0f32, dst_src_size.h as f32)),
        Transform::Flipped270 => {
            Mat3::from_translation(Vec2::new(dst_src_size.h as f32, dst_src_size.w as f32))
        }
    };
    tex_mat = transform_mat * tex_mat;
    tex_mat = translation * tex_mat;

    // now we can add the src crop loc, the size already done implicit by the src size
    tex_mat = Mat3::from_translation(Vec2::new(src.loc.x as f32, src.loc.y as f32)) * tex_mat;

    // at last we have to normalize the values for UV space
    tex_mat = Mat3::from_scale(Vec2::new(
        (1.0f64 / texture.w as f64) as f32,
        (1.0f64 / texture.h as f64) as f32,
    )) * tex_mat;

    tex_mat
}
