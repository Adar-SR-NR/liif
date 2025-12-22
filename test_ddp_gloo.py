import os
import torch
import torch.distributed as dist

def main():
    # 使用 gloo 後端
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='gloo')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Rank {rank}/{world_size} initialized on device {torch.cuda.current_device()} using GLOO")
    
    tensor = torch.tensor([rank], dtype=torch.float32).cuda()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    print(f"Rank {rank} entering all_gather...")
    dist.all_gather(gathered, tensor)
    print(f"Rank {rank} finished all_gather: {[t.item() for t in gathered]}")
    
    dist.destroy_process_group()
    print(f"Rank {rank} done.")

if __name__ == "__main__":
    main()
