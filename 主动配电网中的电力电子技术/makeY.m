function Y = makeY(N, B, num_node, num_branch)

Y=zeros(num_node,num_node);

for k=1:num_branch                    

    I=B(k,1);  

    J=B(k,2);   

    Zs=B(k,7)+j*B(k,8);                %串行(serial)支路阻抗

    if (Zs~=0)

        Ys=1/Zs;

    end

    Yp=j*B(k,9);                       %并行(parallel)支路导纳

    K=B(k,15);

	

	if (B(k,6)==0)                          %若支路为输电线支路

	   Y(I,I)=Y(I,I)+Ys+Yp/2;

	   Y(J,J)=Y(J,J)+Ys+Yp/2;

	   Y(I,J)=Y(I,J)-Ys;

       Y(J,I)=Y(I,J);

	end	   

    

    if (B(k,6)==1)                           %若支路为变压器支路	

       Y(I,I)=Y(I,I)+Ys/K/K;       

       Y(J,J)=Y(J,J)+Ys+Yp;          %采用变压器Π形等效电路

       Y(I,J)=Y(I,J)-Ys/K;           

       Y(J,I)=Y(I,J);

    end

end



for k=1:num_node      %补充节点对地导纳

      Y(k,k)=Y(k,k)+N(k,14)+j*N(k,15);

end

end