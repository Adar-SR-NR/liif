import yaml
import torch
import models
import utils

# Mock config based on the provided one
config_str = """
model:
  name: liif
  args:
    encoder_spec:
      name: arbrcan
      args:
        no_upsampling: true
    imnet_spec:
      name: mlp
      args:
        out_dim: 3
        hidden_list: [256, 256, 256, 256]
"""
config = yaml.safe_load(config_str)

def main():
    print("Creating model...")
    try:
        model = models.make(config['model'])
        print("Model created.")
        n_params = utils.compute_num_params(model, text=True)
        print(f"Number of parameters: {n_params}")
        
        print("\nModel structure:")
        # print(model)
        
        # Check specific parts
        print(f"\nEncoder type: {type(model.encoder)}")
        print(f"Imnet type: {type(model.imnet)}")
        
        params = list(model.parameters())
        print(f"Total parameter tensors: {len(params)}")
        if len(params) > 0:
            print(f"First param shape: {params[0].shape}")
            
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

