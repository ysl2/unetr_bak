
# AntsPath="/media/shuaiw/DataDisk/S_Wang_Research/BRIC-ToolBox/ANTs"
AntsPath="/home/yusongli/.bin/ANTs/install/bin"

# InPath="/media/shuaiw/DataDisk/S_Wang_Research/P_06_HeadNeck_Seg/ProDataset/img_sep_hr"
# OutPath="/media/shuaiw/DataDisk/S_Wang_Research/P_06_HeadNeck_Seg/ProDataset/img_sep_re_hr"
InPath="/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/spacial_input/img"
OutPath="/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/spacial_output/img"

# for folders
# for dir in $(ls "$InPath")
# do
#   if [ -d "$InPath"/"$dir" ]; then
#     echo $dir
#     if [ ! -d "$OutPath"/"$dir" ]; then
#       mkdir "$OutPath"/"$dir"
#     fi
#
#     filelist=`ls "$InPath"/"$dir" `
#     i=0
#     for file in $filelist
#     do
#       echo $file $i
#       i=$((i+1))
#       echo "Doing sample on file" $file
#       ${AntsPath}/ResampleImageBySpacing 3 "$InPath"/"$dir"/"$file" "$OutPath"/"$dir"/"$file" 1 1 3
#     done
#   fi
# done

filelist=`ls $InPath `
i=0
for file in $filelist
do
    echo $file $i
    i=$((i+1))
    echo "Doing sample on file" $file
    ${AntsPath}/ResampleImageBySpacing 3 "$InPath"/"$file" "$OutPath"/"$file" 0.1 0.1 0.5 0 0 1
    # ${AntsPath}/ResampleImage 3 "$InPath"/"$file" "$OutPath"/"$file" 0.6x0.6x3 0 1
done
