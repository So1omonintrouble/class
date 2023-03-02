function Y = makeY(N, B, num_node, num_branch)
Y=zeros(num_node,num_node);
for k=1:num_branch                    
    I=B(k,1);  
    J=B(k,2);   
    Zs=B(k,7)+j*B(k,8);                %����(serial)֧·�迹
    if (Zs~=0)
        Ys=1/Zs;
    end
    Yp=j*B(k,9);                       %����(parallel)֧·����
    K=B(k,15);
	if (B(k,6)==0)                          %��֧·Ϊ�����֧·
	   Y(I,I)=Y(I,I)+Ys+Yp/2;
	   Y(J,J)=Y(J,J)+Ys+Yp/2;
	   Y(I,J)=Y(I,J)-Ys;
       Y(J,I)=Y(I,J);
	end	   

    if (B(k,6)==1)                           %��֧·Ϊ��ѹ��֧·	
       Y(I,I)=Y(I,I)+Ys/K/K;       
       Y(J,J)=Y(J,J)+Ys+Yp;          %���ñ�ѹ�����ε�Ч��·
       Y(I,J)=Y(I,J)-Ys/K;           
       Y(J,I)=Y(I,J);
    end
end


for k=1:num_node      %����ڵ�Եص���
      Y(k,k)=Y(k,k)+N(k,14)+j*N(k,15);
end

end