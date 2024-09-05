use crate::nn::{SeqTypeDim4, SequentialDim4, Upscale, Vgg};
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::{Backend, Tensor};
use burn::tensor::activation::sigmoid;

#[derive(Debug, Module)]
pub struct Unet<B: Backend> {
    vgg: Vgg11Unet<B>,
}

impl<B: Backend> Unet<B> {
    pub fn new(device: &B::Device) -> Self {
        let vgg = Vgg11Unet::new(1, device);
        Self { vgg }
    }

    pub fn forward(&self, tiles: Tensor<B, 4>) -> Tensor<B, 3> {
        let masks = self.vgg.forward(tiles);
        sigmoid(masks).squeeze::<3>(1)
    }
}

#[derive(Debug, Module)]
pub struct Vgg11Unet<B: Backend> {
    encoder: Vec<SequentialDim4<B>>,
    decoder: Vec<Upscale<B>>,
    out: Conv2d<B>,
}

impl<B: Backend> Vgg11Unet<B> {
    pub fn new(out_channels: usize, device: &B::Device) -> Self {
        let vgg: SequentialDim4<B> = Vgg::vgg11(device).features;

        let mut encoder = Vec::new();
        let mut module_layers = Vec::new();

        for layer in vgg.into_iter() {
            if let SeqTypeDim4::MaxPool2d(_) = layer {
                encoder.push(SequentialDim4::from_vec(module_layers));
                module_layers = vec![layer];
            } else {
                module_layers.push(layer);
            }
        }

        let decoder = vec![
            Upscale::new(1024, 256, device),
            Upscale::new(512, 128, device),
            Upscale::new(256, 64, device),
            Upscale::new(128, 64, device),
        ];

        let out = Conv2dConfig::new([64, out_channels], [1, 1]).init(device);

        Self {
            encoder,
            decoder,
            out,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut embeddings = Vec::new();

        let mut x = x;
        for block in &self.encoder {
            embeddings.push(x.clone());
            x = block.forward(x);
        }

        for block in &self.decoder {
            x = block.forward(x, embeddings.pop().unwrap());
        }

        self.out.forward(x)
    }
}
