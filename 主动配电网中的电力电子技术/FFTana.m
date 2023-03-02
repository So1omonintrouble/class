FFTDATA1 = power_fftscope(Ua_B2);
FFTDATA1.fundamental = 50;
FFTDATA1.cycles = 5;
FFTDATA1.freq=0:50:1000;
FFTDATA1.startTime = 1.2;
FFTDATA1 = power_fftscope(FFTDATA1);

FFTDATA2 = power_fftscope(Ua_B5);
FFTDATA2.fundamental = 50;
FFTDATA2.cycles = 5;
FFTDATA2.freq=0:50:1000;
FFTDATA2.startTime = 1.2;
FFTDATA2 = power_fftscope(FFTDATA2);

x = FFTDATA1.freq;
y1 = FFTDATA1.mag;
y2 = FFTDATA2.mag;

% 设置绘图曲线的宽度
lw=1.5;

% 设置绘图的字体大小
ft=10.5; % FontSize

% 声明一个新figure
figure();
tiledlayout(2,1)
ax1=nexttile;
a=bar(x,y1,'LineWidth',1.2);
ax2=nexttile;
b=bar(x,y2,'LineWidth',1.2);
set(a,'edgecolor','none');
set(b,'edgecolor','none');

% 设置x轴范围和标注
xlabel(ax1,'\fontname{宋体}频率\fontname{times new roman}(Hz)','FontSize',ft);
set(ax1,'yscal','log')
set(ax2,'yscal','log')
% 设置y轴范围和标注
ylabel(ax1,'\fontname{宋体}幅值\fontname{times new roman}(%)','FontSize',ft);
legend(ax1,"\fontname{宋体}第一台逆变器\fontname{times new roman}THD=0.01%")
xlim(ax1,[0 1000])
ylim(ax1,[1e-4 1000])
grid(ax1,'on');
ax1.FontSize = ft;
ax1.FontName ="times new roman";
xlabel(ax2,'\fontname{宋体}频率\fontname{times new roman}(Hz)','FontSize',ft);
ylabel(ax2,'\fontname{宋体}幅值\fontname{times new roman}(%)','FontSize',ft);
legend(ax2,"\fontname{宋体}第二台逆变器\fontname{times new roman}THD=0.01%")
xlim(ax2,[0 1000])
ylim(ax2,[1e-4 1000])
grid(ax2,'on');
ax2.FontSize = ft;
ax2.FontName ="times new roman";

% 设置绘图的位置500，500，大小 700宽 300高
set(gcf,'Position',[500,200,600,600]);





