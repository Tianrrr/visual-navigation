## Vision-based Navigation

![alt text](https://github.com/Tianrrr/visual-navigation/blob/main/result/GUI.png?raw=true)
## Thirdparty

`mkdir thirdparty`

Install all dependencies and place them in the "./thirdparty" folder.

You can find [setup instructions here.](wiki/Setup.md)

## Dataset

`mkdir data`

EuRoC MAV Dataset: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

Download data, unzip it, and place it in the "./data" folder.

## Running 

`mkdir build`

`cd build`

`cmake ..`

`make`

`./build/odometry --dataset-path ./data/V1_01_easy/mav0`

## Result

