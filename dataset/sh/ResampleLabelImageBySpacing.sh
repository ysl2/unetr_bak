# AntsPath="/media/shuaiw/DataDisk/S_Wang_Research/BRIC-ToolBox/ANTs"
AntsPath="/home/yusongli/.bin/ANTs/install/bin"

# InPath="/media/shuaiw/DataDisk/S_Wang_Research/P6_Breast_Seg/BreastData/rawdata/relabeling_06112019"
# OutPath="/media/shuaiw/DataDisk/S_Wang_Research/P6_Breast_Seg/BreastData/resampledata/relabeling_06112019"
InPath="/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/spacial_input/mask"
OutPath="/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/spacial_output/mask"

filelist=`ls $InPath `
i=0
for file in $filelist
do
    echo $file $i
    i=$((i+1))
    echo "Doing sample on file" $file
    ${AntsPath}/ResampleImageBySpacing 3 "$InPath"/"$file" "$OutPath"/"$file" 0.1 0.1 0.5 0 0 1
done
