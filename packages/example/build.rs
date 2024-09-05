use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use core::models;

static UNET_TORCH_WEIGHTS: &str = "/home/hazimani/dev/destra-worker/.env/lib/python3.11/site-packages/msintuit_crc/models/weights/v2/brunet-598178c9.pt";

type Backend = burn::backend::NdArray;

fn main() {
    /*  ModelGen::new()
    .input("/home/hazimani/dev/destra-worker/weights/unet_classic3.onnx")
    .out_dir("models/")
    .run_from_script();*/

    // Load weights from torch state_dict
    let device = Default::default();

    let load_args = LoadArgs::new(UNET_TORCH_WEIGHTS.into());
    let record: models::UnetRecord<Backend> = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(load_args, &device)
        .expect("Should load PyTorch model weights");

    // Save the model record to a file.
    let recordert = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    recordert
        .record(record, "unet.mpk".into())
        .expect("Failed to save model record");
}
