# English 

This repository contains the code and data for the article "A Novel Algorithm for Air-Sea Flux Calculation Based on Deep Learning Integrated into CAS-ESM".

## Part One: Python Model Training Code (lhfDPL.py)
This program represents the Torch part of a TBF instance program.

### Step 1: Input Training, Validation, and Test Data

Calculate the required features y from x1,x2,x3...xm, where m = feature_count.

### Step 2: Network Architecture Design:
Define the number of layers and the number of nodes in each layer, as well as the type of activation function.

1.The number of nodes in each layer is set by point_count,n=point_count

2.The number of layers is configured in nn.sequential{}, and the number of activation functions is set later as o.
In nn.sequential, activation functions and linear layers alternate, with ReLU being the commonly used activation function in modern deep learning; Sigmoid, Tanh, etc., can also be used. In Torch, it is best to set them to be the same, and currently, KBF can only set it to one type.

### Step 3: Set Optimization Method,Primarily modify the optimization method, epoch, batch_size, etc., to achieve the best optimization results and obtain optimal parameters  W,b,c,d.

### Step 4: Review Training Results
Generate two graphs showing variance and root mean square error, which help decide if the above steps need to be repeated.

### Step 5: Model Parameter Export
If the training results are satisfactory and correct parameters are obtained, export model parameters to a text file, then use Fortran to read them into TBF.
Check m, n, o, function_kind, shuchucanshu.txt.
Check w_input.txt, w_dense.txt, w_output.txt.
Check b_input.txt, b_dense.txt, b_output.txt.

### Step 6: Transfer *.txt Files
Transfer the generated .txt files to the torch_bridge_fortran folder. The subsequent operations will be handled by the Fortran program.

## Part Two: Fortran Model Code in the TBF Calculation Section
fortran
program main  
The torch_bridge_fortran is written as a module for ease of use in other programs.
Users can treat it as a black box, encapsulating the trained deep learning model inside.
It requires input as a two-dimensional array x_array(cesm_m, ncol); cesm_m is the number of factors, and ncol matches that of CESM. The output y_cesm represents the computational output of the deep model.
```
x1, i --|              ______  
x2, i --|             |     |     
x3, i --|---------->  | box | ------->  y_cesm(i)  
x4, i --|             |_____|  
x5, i --|              
```
This is an example program for calling TBF in CESM. The calculation is performed in the bridge program, with parameter passing handled there.
The parameter optimization process occurs in the Python folder.
The calculation part is mainly written in calculation.f90, and it is not advisable to modify it without a good understanding of the deep model.
File reading happens in file_io.f90.
In the Fortran model code, the following tasks need to be accomplished:

### 1.Use the Module:

```fortran
use bridge , only: tbf  
```
### 2.Declare Variables and Dimensions:

```fortran
implicit none   
integer              :: m = 4  
real, allocatable    :: x_array(:,:)  
real                 :: y_cesm(5)  
character(len = 100) :: dirname ="/data/chengxl/pblh_deeplearning/torch_bridge_fortran/python/"  
```
### 3.Define Independent Variables (can be passed from other functions; if there are multiple variable vectors, array concatenation is necessary):

```fortran
allocate(x_array(m,5))  

x_array(:,1) = (/264.32004, 0.3210011, 14510.625, 52310.562/)  
x_array(:,2) = (/264.31717, 0.32086015, 14449.125, 52227.875/)  
x_array(:,3) = (/264.31717, 0.32067218, 14449.125, 52186.5/)  
x_array(:,4) = (/264.31573, 0.3205077, 14449.125, 52062.375/)  
x_array(:,5) = (/264.31573, 0.3203667, 14387.5, 51979.688/)  
```
### 4.Call Subroutine tbf (passing array lengths, number of factors, independent variable array, and forecast array):

Add functionality to pass a directory parameter so that tbf can be called multiple times.
```fortran
call tbf(dirname, 5, m, x_array, y_cesm)  
```
### Check and Pass Calculation Results:

```fortran
print*, y_cesm   

print*, "Test"

```  


-----------------------------------------------------------------------------------------------------------------------------------------------

# 中文

这个库里存放着文章“A Novel Algorithm for Air-Sea Flux Calculation Based on Deep Learning Integrated into CAS-ESM”的代码和数据。

## 第一部分  python 训练模型代码 lhfDPL.py
这个程序是一个TBF的实例程序的torch部分，

### 第一步：需要输入训练数据，验证数据，测试数据，
        1.计算一个y所需要的x1，x2...xm,这个是m = feature_count .

### 第二步：网络架构设计：设置有几层，每层有几个节点，哪类激活函数
        1.每层有几个节点在point_count,feature,n = point_count 设置；
        2.有几层这个在nn.sequential{}中设置，然后在后面的o = {激活函数的个数} 
        3.在nn.sequential 中激活函数和线性层交替排列，现代深度学习一般relu为激活函数，也可以使sigmod、tanh等，这 
          个在torch最好设为一样的，在KBF目前只能设置成一样的。

### 第三步：设置优化方法，主要是对优化方法，npoch,batch_size,等进行修改，以达到最好的优化效果，得到最优参数W,b,c,d

### 第四步：查看训练的结果，两张图，方差，均方根误差等决定需不需要再次重复上面步骤。

### 第五步：在训练效果很好，得到正确参数的情况下，传输模型参数先输出为txt文件，
        然后用fortran读取到KBF中。
        1.检查m,n,o,function_kind,shuchucanshu.txt
        2.检查w_input.txt w_dense.txt w_output.txt
        3.检查b_input.txt b_dense.txt b_output.txt 

### 第六步：将*.txt 传输到torch_bridge_fortran 文件夹中
         之后的操作留给fortran程序来解决。


## 第二部分  Fortran 模型在模式计算部分 TBF中代码
```fortran
  program main 


!torch_bridge_fortran 被写成了module，方便在其他程序中调用
!其中使用者可以把它当成黑箱，深度学习训练好的模型被封装在其中
!需要输入的是一个二维数组，x_array(cesm_m,ncol),cesm_m 是因子个数
!ncol和cesm的ncol是一致的。输出的y_cesm就是深度模型的计算量

!x1, i --|              ______
!x2, i --|             |     |     
!x3, i --|---------->  | box | ------->  y_cesm(i)
!x4, i --|             |_____|
!x5, i --|              

!这是cesm中调用TBF的一个实例程序，在bridge程序中，计算被进行
!参数传递的过程在bridge进行，
!参数优化的过程则是在python文件夹下进行
!计算过程则主要写在calculation.f90中，这一部分不建议修改，需要对深度模型很了解
!文件读取在file_io.f90中进行
!----------------------------------------------------------------
```
!在fortran模型代码中，我们主要需要做下面几件事


### 1.use module 
```fortran
        use bridge , only: tbf
```        

!-----------------------------------------------------------------

### 2.申明变量和变量的维度
```fortran
        implicit none 
        integer              :: m = 4
        real,allocatable     :: x_array(:,:)
        real                 :: y_cesm(5)
        character(len = 100) :: dirname ="/data/chengxl/&
                         pblh_deeplearning/torch_bridge_fortran/python/"
```
!-----------------------------------------------------------------

### 3.明确自变量（可以通过其他函数传递）!如果多个变量向量则需要进行数组拼接
```fortran
        allocate(x_array(m,5))

        x_array(:,1) = (/264.32004,0.3210011,14510.625,52310.562/)
        x_array(:,2) = (/264.31717,0.32086015,14449.125,52227.875/)
        x_array(:,3) = (/264.31717,0.32067218,14449.125,52186.5/)
        x_array(:,4) = (/264.31573,0.3205077,14449.125,52062.375/)
        x_array(:,5) = (/264.31573,0.3203667,14387.5,51979.688/)
```
!------------------------------------------------------------------

### 4.调用子程序tbf( 数组长度，因子个数，自变量数组, 预报量数组)
!增加功能，传进去一个路径参数使得这个tbf能够被多次调用
```fortran
        call tbf(dirname,5,m,x_array,y_cesm)
```
!------------------------------------------------------------------

### 5.检查和传递计算结果
```fortran
        print*,y_cesm 

        print*,"测试"
```       

