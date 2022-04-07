## Faster depth Estimation for Situational Awareness on Urban Streets

Refer monodepth2 readme for basic instructions

Procedure - 
1. First train model without adding the 'prune' tag. This is the baseline network which is used
for pruning.
   
2. Next prune and fine-tune the backbone model by including 'prune' and 'load_weights_folder'
tags.
   
### Training model
    
    python train.py --model_name <name to save model> --data_path <path to dataset> 

--log_dir - directory to store your model.  
--batch_size - batch size of your data. 
--backbone - encoder model to use. (final model used is mobilenetv2)    
--prune - tag to prune model.    
--load_weights_folder - saved backbone model path used for pruning and fine-tuning.  

### Evaluating model

1. To get performance metrics
    
        python evaluate_depth.py --load_weights_folder <path of the model to evaluate> --data_path <path for the test data> --eval_mono 
   
    --quantization - tag to quantize the model.     
    --obstacle_detection - tag to evaluate at grid level rather than pixel level.   
    --no_cuda - evaluate in cpu mode.   

2. To get inference time and 

        python test_simple.py --image_path <path to test data> --model_name <path to model>

   --quantization - tag to quantize the model.  
   --no_cuda - evaluate in cpu mode.    
   --object_level -  generate depth maps at object level (an entire grid is represented by the 
   min pixel value in that grid)
   
For all the tags that can be used to experiment refer options.py and the respective python files.
