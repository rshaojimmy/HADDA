"""Params for ADDA."""

# params for dataset and data loader

data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
# dataset_mean_value = 0.1307
# dataset_std_value = 0.3081
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 128
image_size = 28
ncls = 10


# params for setting up models
model_root = "snapshots_MU/11"
name = "adaptation3"

# params for source dataset
src_dataset = "MNIST"
src_encoder_restore = "snapshots_MU/11/ADDA-source-encoder-final.pt"
src_classifier_restore = "snapshots_MU/11/ADDA-source-classifier-final.pt"
src_generator_restore = "snapshots_MU/11/ADDA-source-generator-final.pt"
src_model_trained = True

# params for target dataset
tgt_dataset = "USPS"
tgt_encoder_restore = model_root + '/' + name + '/'+ 'ADDA-target-encoder-final.pt'
tgt_classifier_restore  = model_root + '/' + name + '/'+ 'ADDA-target-tgt-classifier-final.pt'

tgt_model_trained = True



inputc = 1
gpuid = [0]
init_type = 'xavier'
g_input_dims = 500
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = model_root + '/' +  name + '/' + 'ADDA-critic-final.pt'


# params for training network
num_gpu = 1
num_epochs_pre_rec = 10
num_epochs_pre = 80
log_step_pre = 40
eval_step_pre = 1
save_step_pre = 20
num_epochs = 200
log_step = 20
save_step = 20
manual_seed = None

#sample
# num_gpu = 1
# num_epochs_pre = 20
# log_step_pre = 10
# eval_step_pre = 1
# save_step_pre = 10
# num_epochs = 100
# log_step = 10
# save_step = 50
# manual_seed = None

# params for optimizing models
learning_rate_pre = 0.001
learning_rate_apt = 0.0002
learning_rate_apt_D = 0.0002
learning_rate_sgd = 0.01
beta1 = 0.5
beta2 = 0.999
momentum = 0.9

para_const = 1
para_autoD = 0.5 
