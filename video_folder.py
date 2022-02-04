import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, nframes, class_to_idx):
    images = []
    n_video = 0
    n_clip = 0
    for target in sorted(os.listdir(dir)):
        if os.path.isdir(os.path.join(dir,target))==True:
            n_video +=1
            # eg: dir + '/rM7aPu9WV2Q'
            subfolder_path = os.path.join(dir, target) 
            for subsubfold in sorted(os.listdir(subfolder_path) ):
                if os.path.isdir(os.path.join(subfolder_path, subsubfold) ):
                	# eg: dir + '/rM7aPu9WV2Q/1'
                    n_clip += 1
                    subsubfolder_path = os.path.join(subfolder_path, subsubfold) 
                    
                    item_frames = []
                    i = 1
                    for fi in sorted( os.listdir(subsubfolder_path) ):
                        if  is_image_file(fi):
                        # fi is an image in the subsubfolder
                            file_name = fi
                            # eg: dir + '/rM7aPu9WV2Q/1/rM7aPu9WV2Q_frames_00086552.jpg'
                            file_path = os.path.join(subsubfolder_path,file_name) 
                            item = (file_path, class_to_idx[target])
                            item_frames.append(item)
                            if i %nframes == 0 and i >0 :
                                images.append(item_frames) # item_frames is a list containing n frames. 
                                item_frames = []
                            i = i+1
    print('number of long videos:')
    print(n_video)
    print('number of short videos')
    print(n_clip)
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    '''
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    '''
    Im = Image.open(path)
    return Im.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
    '''
    return pil_loader(path)


class VideoFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, nframes,  transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, nframes,  class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + 
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.nframes = nframes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # clip is a list of 32 frames 
        clip = self.imgs[index] 
        img_clip = []
        i = 0
        for frame in clip:
            path, target = frame
            img = self.loader(path) 
            i = i+1
            if self.transform is not None:
                img = self.transform(img)
            img = img.view(img.size(0),1, img.size(1), img.size(2)) 
            img_clip.append(img)
        img_frames = torch.cat(img_clip, 1) 
        return img_frames, target

    def __len__(self):
        return len(self.imgs)
