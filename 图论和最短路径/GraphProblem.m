% 两行数据
price = [11  11  12  12  13];
fee = [5  6  8  11  18]; 

%初始邻接矩阵A（n行n列）
n = length(price) + 1; %到年底，即新一年初
A = zeros(n) + Inf;
A(find(diag(ones(1,n)))) = 0;

%用数据给A赋值
for i = 1:n
    for j = i+1:n %对角线之上
        fsum = price(i); %该年初价格
        for tiktok = 1:j-i %每一年里面都要维修，总计经过的年数
            fsum = fsum + fee(tiktok);
        end
        A(i,j) = fsum;
    end
end

%计算单源最短路径
[V,R] = dijkstra(A,1);
%输出最短路径向量
display(V);
%路径分析
M = path2matrix(R);


%邻接矩阵处理
M2 = M;
M2(find(M2 < Inf)) = 1;
M2(find(M2 == Inf)) = 0;
%图的顶点在屏幕上的坐标
XY = [
    1, 2*1.732; 
    3, 2*1.732; 
    4, 1.732;
    3, 0;
    1, 0;
    0, 1.732
    ];
%汇出路径图像和原图
f1 = figure('Name', '原图');
f2 = figure('Name', '顶点1到6的最短路径');
figure(f1);
gplot(A, XY, '-*');
figure(f2);
gplot(M2, XY, '-*');

disp('所需的最小代价为：');
disp(V(6));