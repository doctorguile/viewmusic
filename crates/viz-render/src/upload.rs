//! Per-frame upload of layer geometry and parameters.
//!
//! With [`wgpu::Features::MAPPABLE_PRIMARY_BUFFERS`] (requested by [`crate::device_features`]
//! on integrated GPUs on Metal) the staging belt's mapped chunks are created with
//! vertex/storage/uniform usage and the draws bind them directly, so a frame records no
//! copy commands. Otherwise the belt copies each write into the caller's persistent
//! buffer. Either way chunks are recycled by the belt once the GPU is done with them.

use std::num::NonZeroU64;

use wgpu::util::StagingBelt;

/// Chunk size for the staging belt (256 KiB comfortably holds a typical frame's geometry
/// and uniforms in one reusable mapped buffer).
const CHUNK_BYTES: wgpu::BufferAddress = 256 * 1024;

/// Offset alignment for vertex-buffer uploads.
pub const VERTEX_ALIGN: NonZeroU64 = NonZeroU64::new(wgpu::VERTEX_ALIGNMENT).unwrap();

/// The staging belt plus whether its chunks are bound directly by the draws.
pub struct Uploader {
    belt: StagingBelt,
    direct: bool,
}

impl Uploader {
    pub fn new(device: &wgpu::Device) -> Self {
        let direct = device
            .features()
            .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        let usages = if direct {
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM
        } else {
            wgpu::BufferUsages::COPY_SRC
        };
        Self {
            belt: StagingBelt::new_with_buffer_usages(device.clone(), CHUNK_BYTES, usages),
            direct,
        }
    }

    /// `true` when draws bind the belt's chunks, so modules need no persistent buffer.
    pub fn direct(&self) -> bool {
        self.direct
    }

    /// Reserves `size` bytes for this frame and returns the view to fill plus the buffer
    /// and byte offset the draw must bind.
    ///
    /// `copy_target` is `None` exactly when [`Uploader::direct`]: the bytes then live in a
    /// belt chunk at an `align`-aligned offset. Otherwise the belt copies them into
    /// `copy_target` at `copy_offset`.
    pub fn write(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        copy_target: Option<&wgpu::Buffer>,
        copy_offset: wgpu::BufferAddress,
        size: NonZeroU64,
        align: NonZeroU64,
    ) -> (wgpu::BufferViewMut, wgpu::Buffer, wgpu::BufferAddress) {
        match copy_target {
            Some(target) => (
                self.belt.write_buffer(encoder, target, copy_offset, size),
                target.clone(),
                copy_offset,
            ),
            None => {
                let slice = self.belt.allocate(size, align);
                let buffer = slice.buffer().clone();
                let offset = slice.offset();
                (slice.get_mapped_range_mut(), buffer, offset)
            }
        }
    }

    /// Unmaps this frame's chunks; call before the encoder is submitted.
    pub fn finish(&mut self) {
        self.belt.finish();
    }

    /// Re-maps chunks from submitted frames for reuse; call after submit.
    pub fn recall(&mut self) {
        self.belt.recall();
    }
}

/// Index into `cache` of the bind group for `buffer`, creating it on first sight.
///
/// Direct uploads rotate through a few belt chunks, each needing its own bind group;
/// the set of chunks stops changing once the belt reaches steady state.
pub fn cached_bind_group(
    cache: &mut Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    buffer: &wgpu::Buffer,
    create: impl FnOnce(&wgpu::Buffer) -> wgpu::BindGroup,
) -> usize {
    if let Some(index) = cache.iter().position(|(b, _)| b == buffer) {
        return index;
    }
    cache.push((buffer.clone(), create(buffer)));
    cache.len() - 1
}
