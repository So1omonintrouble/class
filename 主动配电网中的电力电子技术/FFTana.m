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

% ���û�ͼ���ߵĿ��
lw=1.5;

% ���û�ͼ�������С
ft=10.5; % FontSize

% ����һ����figure
figure();
tiledlayout(2,1)
ax1=nexttile;
a=bar(x,y1,'LineWidth',1.2);
ax2=nexttile;
b=bar(x,y2,'LineWidth',1.2);
set(a,'edgecolor','none');
set(b,'edgecolor','none');

% ����x�᷶Χ�ͱ�ע
xlabel(ax1,'\fontname{����}Ƶ��\fontname{times new roman}(Hz)','FontSize',ft);
set(ax1,'yscal','log')
set(ax2,'yscal','log')
% ����y�᷶Χ�ͱ�ע
ylabel(ax1,'\fontname{����}��ֵ\fontname{times new roman}(%)','FontSize',ft);
legend(ax1,"\fontname{����}��һ̨�����\fontname{times new roman}THD=0.01%")
xlim(ax1,[0 1000])
ylim(ax1,[1e-4 1000])
grid(ax1,'on');
ax1.FontSize = ft;
ax1.FontName ="times new roman";
xlabel(ax2,'\fontname{����}Ƶ��\fontname{times new roman}(Hz)','FontSize',ft);
ylabel(ax2,'\fontname{����}��ֵ\fontname{times new roman}(%)','FontSize',ft);
legend(ax2,"\fontname{����}�ڶ�̨�����\fontname{times new roman}THD=0.01%")
xlim(ax2,[0 1000])
ylim(ax2,[1e-4 1000])
grid(ax2,'on');
ax2.FontSize = ft;
ax2.FontName ="times new roman";

% ���û�ͼ��λ��500��500����С 700�� 300��
set(gcf,'Position',[500,200,600,600]);





