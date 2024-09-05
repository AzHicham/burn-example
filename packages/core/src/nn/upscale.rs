use crate::nn::{SeqTypeDim4, SequentialDim4};
use burn::module::Module;
use burn::nn::conv::{Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{PaddingConfig2d, Relu};
use burn::prelude::{Backend, Tensor};

#[derive(Debug, Module)]
pub struct Upscale<B: Backend> {
    up: ConvTranspose2d<B>,
    conv: SequentialDim4<B>,
}

impl<B: Backend> Upscale<B> {
    pub fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let up = ConvTranspose2dConfig::new([in_channels / 2, in_channels / 2], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let conv = SequentialDim4::from_vec(vec![
            SeqTypeDim4::Conv2d(
                Conv2dConfig::new([in_channels, out_channels], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            ),
            SeqTypeDim4::Relu(Relu::new()),
            SeqTypeDim4::Conv2d(
                Conv2dConfig::new([out_channels, out_channels], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            ),
            SeqTypeDim4::Relu(Relu::new()),
        ]);

        Self { up, conv }
    }

    pub fn forward(&self, x1: Tensor<B, 4>, x2: Tensor<B, 4>) -> Tensor<B, 4> {
        let x1 = self.up.forward(x1);
        let (w_padding, h_padding) = (x1.dims()[2] - x2.dims()[2], x1.dims()[3] - x2.dims()[3]);
        let x2 = x2.pad(
            (w_padding / 2, w_padding / 2, h_padding / 2, h_padding / 2),
            <B as Backend>::FloatElem::default(),
        );
        let x = Tensor::<B, 4>::cat(vec![x1, x2], 1);
        self.conv.forward(x)
    }
}
