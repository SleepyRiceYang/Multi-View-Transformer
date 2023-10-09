import os 

def set_gpu(gpu_input):
    # import pdb;pdb.set_trace()
    # if gpu_input == 'all':
    #     gpus = get_device_id()
    # else:
    #     gpus = gpu_input
    gpus = gpu_input
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))