use llama_rs::LoadProgress;

//see https://github.com/rustformers/llama-rs/blob/main/llama-cli/src/cli_args.rs
pub fn log_load_progress(progress: LoadProgress) {
    match progress {
        LoadProgress::HyperparametersLoaded(hparams) => {
            println!("Loaded hyperparameters {hparams:#?}")
        }
        LoadProgress::ContextSize { bytes } => println!(
            "ggml ctx size = {:.2} MB\n",
            bytes as f64 / (1024.0 * 1024.0)
        ),
        LoadProgress::PartLoading {
            file,
            current_part,
            total_parts,
        } => {
            let current_part = current_part + 1;
            println!(
                "Loading model part {}/{} from '{}'\n",
                current_part,
                total_parts,
                file.to_string_lossy(),
            )
        }
        LoadProgress::PartTensorLoaded {
            current_tensor,
            tensor_count,
            ..
        } => {
            let current_tensor = current_tensor + 1;
            if current_tensor % 8 == 0 {
                println!("Loaded tensor {current_tensor}/{tensor_count}");
            }
        }
        LoadProgress::PartLoaded {
            file,
            byte_size,
            tensor_count,
        } => {
            println!("Loading of '{}' complete", file.to_string_lossy());
            println!(
                "Model size = {:.2} MB / num tensors = {}",
                byte_size as f64 / 1024.0 / 1024.0,
                tensor_count
            );
        }
    }
}
