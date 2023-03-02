y1 = f1(:,2);
y2 = f2(:,2);
t=f1(:,1);
% 设置绘图曲线的宽度
lw=2;

% 设置绘图的字体大小
ft=10.5; % FontSize

% 声明一个新figure
figure();
tiledlayout(2,1)
ax1=nexttile;

plot(t,y1,'LineWidth',lw);

ax2=nexttile;
plot(t,y2,'LineWidth',lw);
% 设置x轴范围和标注
xlabel(ax1,'\fontname{宋体}仿真时间\fontname{times new roman}(s)','FontSize',ft);
ylabel(ax1,'\fontname{宋体}第一台逆变器频率\fontname{times new roman}(Hz)','FontSize',ft);
%legend(ax1,"\fontname{宋体}第一台逆变器输出有功功率\fontname{times new roman}(W)")
xlim(ax1,[0 1.5])
ylim(ax1,[49.9,50.1])
grid(ax1,'on');
ax1.FontSize = ft;
ax1.FontName ="times new roman";

xlabel(ax2,'\fontname{宋体}仿真时间\fontname{times new roman}(s)','FontSize',ft);
ylabel(ax2,'\fontname{宋体}第二台逆变器频率\fontname{times new roman}(Hz)','FontSize',ft);
%legend(ax2,"\fontname{宋体}第二台逆变器输出有功功率\fontname{times new roman}(W)")
xlim(ax2,[0 1.5])
ylim(ax2,[49.9,50.1])
grid(ax2,'on');
ax2.FontSize = ft;
ax2.FontName ="times new roman";

% 设置绘图的位置500，500，大小 700宽 300高
set(gcf,'Position',[700,50,600,600]);