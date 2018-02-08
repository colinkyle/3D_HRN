using Images
using ImageView
# set data dir
main_dir = "/Users/colin/Dropbox/__Atlas__/"
data_dir = "/Users/colin/Dropbox/__Atlas__/data"
# get list of subject folders
cd(data_dir)
sub_dirs = filter(x->isdir(x),readdir(data_dir))

## Subject loop starts here
s = 1
# cd into subfolders
cd(abspath(joinpath(sub_dirs[1],"histology/")))
# get image names
imgnames = filter(x->contains(x,".tif"), readdir())

## Between section registration starts here
fixi = 1
movi = 2
# load images
fixed = separate(load(imgnames[fixi]))
#using Winston
#imagesc(data(fixed))

#fixed = convert(Float32,load(imgnames[fixi]))
## start function
   degree = 10.0
   rotated = fixed
   # rotate image
   angle = degree*pi/180.0
   #pre allocate dimensions
   # calc rotation
   for i in 1:size(fixed,1)
   #i=2
   #j=1
   for j in 1:size(fixed,2)
         x = (i-1)*cos(angle) + (j-1)*sin(angle)
         y = -(i-1)*sin(angle) + (j-1)*cos(angle)
         x = round(x)+1
         y = round(y)+1
         if x>0 && y >0 && x<=size(fixed,1) && y<=size(fixed,2)
         rotated.data[i,j] = fixed.data[Int64(x),Int64(y)]
      end
   end
   end
   #return rotated

#fixed = imrotate(fixed,10.0)
ImageView.view(fixed)
ImageView.view(rotated)
