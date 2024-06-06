import os
import h5py
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "datasets", "simpler")

def main():
    # file = h5py.File(os.path.join(DATA_DIR, "ep_2024-06-02_184950.h5"), "r")
        
    # steps = []
    # for step in file['steps']:
    #     steps.append(step)
        
    # steps.sort(key = lambda x: int(x.split('_')[1]))
    # print(len(steps))
    
    # for step in steps:
    #     print(step)
    #     plt.imshow(file['steps'][step]['observation']['simpler-img'][:])
    #     plt.pause(0.1)
        
    file = h5py.File(os.path.join(DATA_DIR, "dataset.h5"), "r")
    
    def print_tree(name, group):
        if isinstance(group, h5py.Group):
            print(name + "/")
        else:
            print(name)
                
    file.visititems(print_tree)
        
    
if __name__ == "__main__":
    main()