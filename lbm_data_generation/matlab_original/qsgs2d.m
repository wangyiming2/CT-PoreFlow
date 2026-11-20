clear all
clc

lx =300;         %尺寸    %白色为基质，黑色为孔隙，生长相视为基质值为1，非生长相视为孔隙值为0
ly =300;

d1 = 0.01; d2 = 0.01; d3 = 0.01; d4 = 0.01; d7 = 0.03; d8 = 0.03;             %各方向生长概率，与三维方向一致，因此未改成 5 6 7 8
d9 = 0.03; d10 = 0.03; 

n = 0.6;%孔隙率
cdd = 0.005;%种子概率
axis_ =  1 ;%如果等于1 则提取连通域
numtotal_need= (1-n) * lx * ly ;%固体所需数量
numsolid = 0;
arrgrid=zeros(lx,ly);                     
solid = zeros;

% 四参数随机生成   8个方向
indices = randperm(lx*ly);
for index = 1:(lx*ly)
    [i, j] = ind2sub([lx, ly], indices(index));
    if rand() <  cdd  % && numsolid < 20   %取消百分号并设置值可控制种子数目
        numsolid= numsolid+ 1;
        arrgrid(i, j) = 1;
        solid(numsolid, 1) = i;
        solid(numsolid, 2) = j;
    end
end
disp(['种子数目: ' num2str(numsolid)]);
tnumsolid= numsolid;   %固体种子数量
if  tnumsolid == 0
    disp('固体种子数目为0，提高生长核概率再运行')
    return;
end
while tnumsolid<= numtotal_need
    for index_solid= 1: tnumsolid
        index_i= solid(index_solid, 1);
        index_j= solid(index_solid, 2);
        if tnumsolid>=numtotal_need       
            break
        end
        %向 1 方向生长
        if index_i< lx
            i= index_i+1; j = index_j;         %(i+1, j, k) 
            if (arrgrid(i,j)== 0 && rand( ) < d1)
                numsolid= numsolid+ 1; arrgrid(i,j) = 1;
                solid( numsolid, 1) = i; solid(numsolid, 2) = j; 
                tnumsolid=numsolid;
            end
        end
        if tnumsolid>=numtotal_need
            break
        end
        %向 2 方向生长
        if index_j< ly
            i= index_i; j = index_j + 1;          % （i,j+1,k）
            if (arrgrid(i,j)== 0 && rand( ) < d2)
                numsolid= numsolid+ 1; arrgrid(i,j) = 1;
                solid( numsolid, 1) = i; solid(numsolid, 2) = j; 
                tnumsolid=numsolid;
            end
        end
        if tnumsolid>=numtotal_need
            break
        end
        %向 3 方向生长
        if index_i> 1
            i= index_i-1; j = index_j;         %（i-1,j,k）
            if (arrgrid(i,j)== 0 && rand( ) < d3)
                numsolid= numsolid+ 1; arrgrid(i,j) = 1;
                solid( numsolid, 1) = i; solid(numsolid, 2) = j; 
                tnumsolid=numsolid;
            end
        end
        if tnumsolid>=numtotal_need
            break
        end
        %向 4 方向生长
        if index_j> 1
            i= index_i; j = index_j-1;                  %（i,j-1,k）
            if (arrgrid(i,j)== 0 && rand( ) < d4)
                numsolid= numsolid+ 1; arrgrid(i,j) = 1;
                solid( numsolid, 1) = i; solid(numsolid, 2) = j; 
                tnumsolid=numsolid;
            end
        end
        if tnumsolid>=numtotal_need
            break
        end
       
        %向 5 方向生长
        if index_i< lx && index_j< ly
            i= index_i+1; j = index_j+1;                    %（i+1,j+1,k）
            if (arrgrid(i,j)== 0 && rand( ) < d7)
                numsolid= numsolid+ 1; arrgrid(i,j) = 1;
                solid( numsolid, 1) = i; solid(numsolid, 2) = j; 
                tnumsolid=numsolid;
            end
        end
        if tnumsolid>=numtotal_need
            break
        end
        %向 6 方向生长
        if index_i> 1 && index_j< ly
            i= index_i-1; j = index_j+1;                    %%（i-1,j+1,k）
            if (arrgrid(i,j)== 0 && rand( ) < d8)
                numsolid= numsolid+ 1; arrgrid(i,j) = 1;
                solid( numsolid, 1) = i; solid(numsolid, 2) = j; 
                tnumsolid=numsolid;
            end
        end
        if tnumsolid>=numtotal_need
            break
        end
        %向 7 方向生长
        if index_i> 1 && index_j> 1
            i= index_i-1; j = index_j-1;                    %（i-1,j-1,k）
            if (arrgrid(i,j)== 0 && rand( ) < d9)
                numsolid= numsolid+ 1; arrgrid(i,j) = 1;
                solid( numsolid, 1) = i; solid(numsolid, 2) = j; 
                tnumsolid=numsolid;
            end
        end
        if tnumsolid>=numtotal_need
            break
        end
        %向 8 方向生长
        if index_i< lx && index_j> 1
            i= index_i+1; j = index_j-1;            %（i+1,j-1,k）
            if (arrgrid(i,j)== 0 && rand( ) < d10)
                numsolid= numsolid+ 1; arrgrid(i,j) = 1;
                solid( numsolid, 1) = i; solid(numsolid, 2) = j; 
                tnumsolid=numsolid;
            end
        end
        if tnumsolid>=numtotal_need
            break
        end      
    end
    if tnumsolid >= numtotal_need
        break
    end
end


psolid = sum(arrgrid(:)==1)/(lx*ly);
disp(['生长完后验证孔隙率为 ' num2str(1 - psolid)]);


for  ii = 1:3  %中值滤波迭代次数
aaas = 3; % 中值滤波核尺寸     用不上就ctrl+R注释掉
arrgrid = medfilt2(arrgrid, [aaas, aaas]);  %因为用到了中值滤波所以会对原数组arrgrid产生影响，数组arrgrid孔隙率会发生变化
end



if  axis_ == 1
%去除孤立区域     %这一版是直接原数组arrgrid中去除孤立的
AAA = 0;%提取0的连通域  
BBB = 8; %采用8连通  4
cc = bwconncomp(arrgrid==AAA, BBB);  
pixel_lists = cc.PixelIdxList;
num_regions = cc.NumObjects;  
disp(['去除前共检测到该值有 ' num2str(num_regions) ' 个区域']);
for region_id = 1:num_regions  
    pixel_list = pixel_lists{region_id};
    [x, y] = ind2sub(size(arrgrid), pixel_list);
    if any(y == ly) && any(y == 1)%列方向
        %             if any(x == lx) && any(x == 1)  %行方向
        arrgrid(pixel_list) = AAA;
    else                           
        arrgrid(pixel_list) = 1-AAA;%如果不和进出口连接就把他变成另外一相
    end
end

num_select = sum(arrgrid(:)== AAA);%记录该相的数量
arrgrid = double(arrgrid);
%%
%总连通率
P_select = num_select/(lx*ly);   
cc = bwconncomp(arrgrid==AAA, BBB);  %2的连通性
num_regions = cc.NumObjects; 
disp(['去除后共检测到该值有 ' num2str(num_regions) ' 个连通区域,占比为：',num2str(P_select),' 连通数组为arrgrid']);
end
%%

psolid_end = sum(arrgrid(:)==1)/(lx*ly);
disp(['最终孔隙率为 ' num2str(1 - psolid_end)]);

%画图
imagesc(arrgrid);colormap(gray);axis on; xlabel('X 轴');ylabel('Y 轴');  
% save('arrgrid.mat', 'arrgrid');
%保存tif
% random = randi([1, 10000]);
% filename = sprintf('result\\image%d.tif', random);
% saveas(gcf, filename, 'tiffn');


