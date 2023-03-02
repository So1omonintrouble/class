y1a = Uabc_B2(:,2);
y1b = Uabc_B2(:,3);
y1c = Uabc_B2(:,4);
y2a = Uabc_B5(:,2);
y2b = Uabc_B5(:,3);
y2c = Uabc_B5(:,4);
t=Uabc_B2(:,1);
% 设置绘图曲线的宽度
lw=2;

% 设置绘图的字clear体大小
ft=10.5; % FontSize

% 声明一个新figure
figure();
tiledlayout(2,1)
ax1=nexttile;

plot(t,y1a,'LineWidth',lw);
hold on
plot(t,y1b,'LineWidth',lw);
plot(t,y1c,'LineWidth',lw);
hold off
ax2=nexttile;
plot(t,y2a,'LineWidth',lw);
hold on
plot(t,y2b,'LineWidth',lw);
plot(t,y2c,'LineWidth',lw);
hold off
% 设置x轴范围和标注
xlabel(ax1,'\fontname{宋体}仿真时间\fontname{times new roman}(s)','FontSize',ft);
ylabel(ax1,'\fontname{宋体}第一台逆变器输出三相电压\fontname{times new roman}(V)','FontSize',ft);
legend(ax1,"\fontname{times new roman}A\fontname{宋体}相电压","\fontname{times new roman}B\fontname{宋体}相电压","\fontname{times new roman}C\fontname{宋体}相电压",'FontSize',ft)
xlim(ax1,[1.2 1.3])
ylim(ax1,[-400,400])
grid(ax1,'on');
ax1.FontSize = ft;
ax1.FontName ="times new roman";

xlabel(ax2,'\fontname{宋体}仿真时间\fontname{times new roman}(s)','FontSize',ft);
ylabel(ax2,'\fontname{宋体}第二台逆变器输出三相电压\fontname{times new roman}(V)','FontSize',ft);
legend(ax2,"\fontname{times new roman}A\fontname{宋体}相电压","\fontname{times new roman}B\fontname{宋体}相电压","\fontname{times new roman}C\fontname{宋体}相电压",'FontSize',ft)
%legend(ax2,"\fontname{宋体}第二台逆变器输出有功功率\fontname{times new roman}(W)")
xlim(ax2,[1.2 1.3])
ylim(ax2,[-400,400])
grid(ax2,'on');
ax2.FontSize = ft;
ax2.FontName ="times new roman";

% 设置绘图的位置500，500，大小 700宽 300高
set(gcf,'Position',[700,50,600,600]);