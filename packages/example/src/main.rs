use burn::backend::libtorch::LibTorchDevice;
use burn::backend::LibTorch;
use burn::module::Module;
use burn::prelude::{Backend, Tensor};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_example::models::Unet;

fn main() {
    let device = LibTorchDevice::Cuda(0);
    run_unet::<LibTorch>(0, device);
}

fn run_unet<B: Backend>(iter: usize, device: B::Device) {
    let model: Unet<B> = Unet::new(&device);

    // `load_record` takes ownership of the model but we can re-assign the returned value
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .load("./weights/unet.mpk".into(), &device)
        .expect("Should decode state successfully");

    let model = model.load_record(record);

    let start = std::time::Instant::now();

    for _ in 0..iter {
        let x = Tensor::empty([4, 3, 512, 512], &device);
        model.forward(x);
    }

    let elapsed = start.elapsed();
    println!("{:?}", elapsed);

    let y = model.forward(Tensor::empty([4, 3, 512, 512], &device));
    dbg!(y.shape());
}
