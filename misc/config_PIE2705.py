"""Params for ADDA."""

# params for dataset and data loader

data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 64
image_size = 32
ncls = 68

# params for setting up models
model_root = "snapshots_PIE/05sm"
name = "adaptation_const0"

# params for source dataset
src_dataset = "PIE27"
src_encoder_restore = "snapshots_PIE/05sm/ADDA-source-encoder-final.pt"
src_classifier_restore = "snapshots_PIE/05sm/ADDA-source-classifier-final.pt"
src_model_trained = True

# params for target dataset
tgt_dataset = "PIE05"
tgt_encoder_restore = "snapshots_PIE/05sm/ADDA-source-encoder-final.pt"
tgt_classifier_restore = model_root + '/' + name + '/'+ 'ADDA-target-tgt-classifier-final.pt'
tgt_model_trained = True



# d_input_dims = 512

inputc = 1
g_input_dims = 500
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = "snapshots_PIE/05sm/ADDA-critic-final.pt"

# generator_restore = "snapshots/svhn_mnist1/ADDA-generator-final.pt"
# discriminator_restore = "snapshots/svhn_mnist1/ADDA-discriminator-final.pt"

# params for training network
init_type = 'xavier'
gpuid = [0]
num_gpu = 1
num_epochs_pre = 40
log_step_pre = 50
eval_step_pre = 1
save_step_pre = 2
num_epochs = 400
log_step = 20
save_step = 20
manual_seed = None

# params for optimizing models
learning_rate_pre = 0.001
learning_rate_apt = 0.0002
learning_rate_apt_D = 0.0002
learning_rate_sgd = 0.01
beta1 = 0.5
beta2 = 0.999
momentum = 0.9

para_const = 0
para_autoD = 0.5 
