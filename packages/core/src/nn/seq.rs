use burn::module::Module;
use burn::nn::conv::Conv2d;
use burn::nn::pool::MaxPool2d;
use burn::nn::{BatchNorm, Dropout, Linear, Relu};
use burn::prelude::Backend;
use burn::tensor::Tensor;

#[derive(Debug, Module)]
pub enum SeqType<B: Backend> {
    Dim2(SeqTypeDim2<B>),
    Dim4(SeqTypeDim4<B>),
}

#[derive(Debug, Module)]
pub enum SeqTypeDim2<B: Backend> {
    BatchNorm(BatchNorm<B, 4>),
    // Sequential(SequentialDim2<B>),
    Relu(Relu),
    Linear(Linear<B>),
    Dropout(Dropout),
}

#[derive(Debug, Module)]
pub struct SequentialDim2<B: Backend> {
    layers: Vec<SeqTypeDim2<B>>,
}

impl<B: Backend> SequentialDim2<B> {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn from_vec(layers: Vec<SeqTypeDim2<B>>) -> Self {
        Self { layers }
    }

    pub fn iter(&self) -> std::slice::Iter<SeqTypeDim2<B>> {
        self.layers.iter()
    }

    pub fn into_iter(self) -> std::vec::IntoIter<SeqTypeDim2<B>> {
        self.layers.into_iter()
    }

    pub fn add(&mut self, layer: SeqTypeDim2<B>) {
        self.layers.push(layer);
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = x;
        for layer in &self.layers {
            match layer {
                SeqTypeDim2::BatchNorm(bn) => x = bn.forward(x),
                //  SeqTypeDim2::Sequential(sequential) => x = sequential.forward(x),
                SeqTypeDim2::Relu(relu) => x = relu.forward(x),
                SeqTypeDim2::Linear(lin) => x = lin.forward(x),
                SeqTypeDim2::Dropout(dropout) => x = dropout.forward(x),
            }
        }
        x
    }
}

impl<B: Backend> Iterator for SequentialDim2<B> {
    type Item = SeqTypeDim2<B>;

    fn next(&mut self) -> Option<Self::Item> {
        self.layers.pop()
    }
}

#[derive(Debug, Module)]
pub enum SeqTypeDim4<B: Backend> {
    Conv2d(Conv2d<B>),
    BatchNorm(BatchNorm<B, 4>),
    MaxPool2d(MaxPool2d),
    // Sequential(SequentialDim4<B>),
    Relu(Relu),
    Linear(Linear<B>),
    Dropout(Dropout),
}

#[derive(Debug, Module)]
pub struct SequentialDim4<B: Backend> {
    layers: Vec<SeqTypeDim4<B>>,
}

impl<B: Backend> SequentialDim4<B> {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn from_vec(layers: Vec<SeqTypeDim4<B>>) -> Self {
        Self { layers }
    }

    pub fn iter(&self) -> impl Iterator<Item = &SeqTypeDim4<B>> {
        self.layers.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = SeqTypeDim4<B>> {
        self.layers.into_iter()
    }

    pub fn add(&mut self, layer: SeqTypeDim4<B>) {
        self.layers.push(layer);
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;
        for layer in &self.layers {
            match layer {
                SeqTypeDim4::Conv2d(conv) => x = conv.forward(x),
                SeqTypeDim4::BatchNorm(bn) => x = bn.forward(x),
                //  SeqTypeDim4::Sequential(sequential) => x = sequential.forward(x),
                SeqTypeDim4::Relu(relu) => x = relu.forward(x),
                SeqTypeDim4::MaxPool2d(pool) => x = pool.forward(x),
                SeqTypeDim4::Linear(lin) => x = lin.forward(x),
                SeqTypeDim4::Dropout(dropout) => x = dropout.forward(x),
            }
        }
        x
    }
}
