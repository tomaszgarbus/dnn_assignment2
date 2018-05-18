mkdir assignment2_resized
mkdir assignment2_resized/training
mkdir assignment2_resized/training/images
mkdir assignment2_resized/training/labels_plain

new_size=$1
echo "Resizing to size $new_size"

echo "Images"
i=0
for l in `ls assignment2/training/images`;
do
  i=$((i+1))
  echo $i
  convert -resize $new_size assignment2/training/images/$l\
    assignment2_resized/training/images/$l;
done

echo "Labels plain"
i=0
for l in `ls assignment2/training/labels_plain`;
do
  i=$((i+1))
  echo $i
  convert -resize $new_size assignment2/training/labels_plain/$l\
    assignment2_resized/training/labels_plain/$l;
done
