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
    /// and `align`-aligned byte offset the draw must bind.
    ///
    /// `copy_target` is `None` exactly when [`Uploader::direct`]: the bytes then live in a
    /// belt chunk. Otherwise the belt copies them into `copy_target` at its cursor, which
    /// then advances by `size` rounded up to `align`.
    pub fn write(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        copy_target: Option<&mut CopyTarget>,
        size: NonZeroU64,
        align: NonZeroU64,
    ) -> (wgpu::BufferViewMut, wgpu::Buffer, wgpu::BufferAddress) {
        match copy_target {
            Some(target) => {
                let offset = target.cursor;
                target.cursor += size.get().next_multiple_of(align.get());
                (
                    self.belt
                        .write_buffer(encoder, &target.buffer, offset, size),
                    target.buffer.clone(),
                    offset,
                )
            }
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

/// A persistent buffer that uploads are copied into when they are not bound directly,
/// with this frame's running write offset.
pub struct CopyTarget {
    buffer: wgpu::Buffer,
    /// Byte offset of the next [`Uploader::write`] into `buffer`.
    cursor: wgpu::BufferAddress,
}

impl CopyTarget {
    pub fn new(buffer: wgpu::Buffer) -> Self {
        Self { buffer, cursor: 0 }
    }

    /// Makes the next write start at the beginning of the buffer again.
    pub fn rewind(&mut self) {
        self.cursor = 0;
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
