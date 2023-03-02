y1 = P1(:,2);
y2 = Q1(:,2);
y3 = P2(:,2);
y4 = Q2(:,2);
t=P1(:,1);
% ���û�ͼ���ߵĿ��
lw=2;

% ���û�ͼ�������С
ft=10.5; % FontSize

% ����һ����figure
figure();
tiledlayout(2,1)
ax1=nexttile;

plot(t,y1,'LineWidth',lw);
hold on
plot(t,y2,'LineWidth',lw);
ax2=nexttile;
plot(t,y3,'LineWidth',lw);
hold on
plot(t,y4,'LineWidth',lw);
hold off
% ����x�᷶Χ�ͱ�ע
xlabel(ax1,'\fontname{����}����ʱ��\fontname{times new roman}(s)','FontSize',ft);
ylabel(ax1,'\fontname{����}��һ̨������������\fontname{times new roman}','FontSize',ft);
legend(ax1,"\fontname{����}��һ̨���������й�����\fontname{times new roman}(W)","\fontname{����}��һ̨���������޹�����\fontname{times new roman}(Var)")
xlim(ax1,[0 1.5])
ylim(ax1,[0,10000])
grid(ax1,'on');
ax1.FontSize = ft;
ax1.FontName ="times new roman";

xlabel(ax2,'\fontname{����}����ʱ��\fontname{times new roman}(s)','FontSize',ft);
ylabel(ax2,'\fontname{����}�ڶ�̨������������\fontname{times new roman}','FontSize',ft);
legend(ax2,"\fontname{����}�ڶ�̨���������й�����\fontname{times new roman}(W)","\fontname{����}�ڶ�̨���������޹�����\fontname{times new roman}(Var)")
xlim(ax2,[0 1.5])
ylim(ax2,[0,10000])
grid(ax2,'on');
ax2.FontSize = ft;
ax2.FontName ="times new roman";

% ���û�ͼ��λ��500��500����С 700�� 300��
set(gcf,'Position',[700,50,600,600]);





