## Visual-Inertial Odometry

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

## Result (Comparison with Visual Odometry)
![alt text](https://github.com/Tianrrr/visual-navigation/blob/main/result/WechatIMG103.jpg?raw=true)

### MH_01
<div align=center>
  <img src=https://github.com/Tianrrr/visual-navigation/blob/main/result/mh_01/vo_xyz.png width = 50%><img src=https://github.com/Tianrrr/visual-navigation/blob/main/result/mh_01/vio_xyz.png width = 50%>
  <p>left: vo; right: vio</p>
</div>

### MH_05
<div align=center>
  <img src=https://github.com/Tianrrr/visual-navigation/blob/main/result/mh_05/vo_xyz.png width = 50%><img src=https://github.com/Tianrrr/visual-navigation/blob/main/result/mh_05/vio_xyz.png width = 50%>
  <p>left: vo failed; right: vio</p>
</div>




