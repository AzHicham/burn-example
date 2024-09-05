use burn::module::Module;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2dConfig};
use burn::nn::{BatchNormConfig, DropoutConfig, LinearConfig, PaddingConfig2d, Relu};
use burn::prelude::{Backend, Tensor};

use super::seq::{SeqTypeDim2, SeqTypeDim4, SequentialDim2, SequentialDim4};

#[derive(Debug, Module)]
pub struct Vgg<B: Backend> {
    pub(crate) features: SequentialDim4<B>,
    avgpool: AdaptiveAvgPool2d,
    classifier: SequentialDim2<B>,
}

impl<B: Backend> Vgg<B> {
    fn new(
        key: VggKey,
        num_classes: usize,
        dropout: f64,
        batch_norm: bool,
        device: &B::Device,
    ) -> Self {
        let features: SequentialDim4<B> = make_features(&get_cfg(key), batch_norm, device);
        let avgpool = AdaptiveAvgPool2dConfig::new([7, 7]).init();
        let classifier: SequentialDim2<B> = make_classifier(device, num_classes, dropout);
        Self {
            features,
            avgpool,
            classifier,
        }
    }

    pub fn vgg11(device: &B::Device) -> Self {
        Self::new(VggKey::A, 1000, 0.5, false, device)
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.features.forward(x);
        let x = self.avgpool.forward(x);
        let x: Tensor<B, 2> = x.flatten(1, 3);
        let x = self.classifier.forward(x);
        x
    }
}

pub enum VggKey {
    A,
    B,
    D,
    E,
}

enum LayerConfig {
    MaxPool,
    Channels(usize),
}

fn make_features<B: Backend>(
    cfg: &[LayerConfig],
    batch_norm: bool,
    device: &B::Device,
) -> SequentialDim4<B> {
    let mut layers = SequentialDim4::new();
    let mut in_channels = 3;

    for v in cfg {
        match v {
            LayerConfig::MaxPool => {
                let max_pool = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
                layers.add(SeqTypeDim4::MaxPool2d(max_pool));
            }
            LayerConfig::Channels(out_channels) => {
                let conv2d = Conv2dConfig::new([in_channels, *out_channels], [3, 3])
                    .with_stride([1, 1])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .with_bias(true)
                    .init(device);
                if batch_norm {
                    let batch_norm = BatchNormConfig::new(*out_channels).init(device);
                    layers.add(SeqTypeDim4::Conv2d(conv2d));
                    layers.add(SeqTypeDim4::BatchNorm(batch_norm));
                    layers.add(SeqTypeDim4::Relu(Relu::new()));
                } else {
                    layers.add(SeqTypeDim4::Conv2d(conv2d));
                    layers.add(SeqTypeDim4::Relu(Relu::new()));
                }
                in_channels = *out_channels;
            }
        }
    }
    layers
}

fn make_classifier<B: Backend>(
    device: &B::Device,
    num_class: usize,
    dropout: f64,
) -> SequentialDim2<B> {
    let mut classifier = SequentialDim2::new();
    classifier.add(SeqTypeDim2::Linear(
        LinearConfig::new(512 * 7 * 7, 4096).init(device),
    ));
    classifier.add(SeqTypeDim2::Relu(Relu::new()));
    classifier.add(SeqTypeDim2::Dropout(DropoutConfig::new(dropout).init()));
    classifier.add(SeqTypeDim2::Linear(
        LinearConfig::new(4096, 4096).init(device),
    ));
    classifier.add(SeqTypeDim2::Relu(Relu::new()));
    classifier.add(SeqTypeDim2::Dropout(DropoutConfig::new(dropout).init()));
    classifier.add(SeqTypeDim2::Linear(
        LinearConfig::new(4096, num_class).init(device),
    ));
    classifier
}

fn get_cfg(key: VggKey) -> Vec<LayerConfig> {
    match key {
        VggKey::A => vec![
            LayerConfig::Channels(64),
            LayerConfig::MaxPool,
            LayerConfig::Channels(128),
            LayerConfig::MaxPool,
            LayerConfig::Channels(256),
            LayerConfig::Channels(256),
            LayerConfig::MaxPool,
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::MaxPool,
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::MaxPool,
        ],
        VggKey::B => vec![
            LayerConfig::Channels(64),
            LayerConfig::Channels(64),
            LayerConfig::MaxPool,
            LayerConfig::Channels(128),
            LayerConfig::Channels(128),
            LayerConfig::MaxPool,
            LayerConfig::Channels(256),
            LayerConfig::Channels(256),
            LayerConfig::MaxPool,
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::MaxPool,
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::MaxPool,
        ],
        VggKey::D => vec![
            LayerConfig::Channels(64),
            LayerConfig::Channels(64),
            LayerConfig::MaxPool,
            LayerConfig::Channels(128),
            LayerConfig::Channels(128),
            LayerConfig::MaxPool,
            LayerConfig::Channels(256),
            LayerConfig::Channels(256),
            LayerConfig::Channels(256),
            LayerConfig::MaxPool,
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::MaxPool,
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::MaxPool,
        ],
        VggKey::E => vec![
            LayerConfig::Channels(64),
            LayerConfig::Channels(64),
            LayerConfig::MaxPool,
            LayerConfig::Channels(128),
            LayerConfig::Channels(128),
            LayerConfig::MaxPool,
            LayerConfig::Channels(256),
            LayerConfig::Channels(256),
            LayerConfig::Channels(256),
            LayerConfig::Channels(256),
            LayerConfig::MaxPool,
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::MaxPool,
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::Channels(512),
            LayerConfig::MaxPool,
        ],
    }
}
