function [ Jacobi ] = Jacobi_PQ(bus,node,V,B)
for ii=1:bus
    if node(ii,1)~=1
        for j=1:bus
            if node(j,1)~=1
                if ii~=j
                   H(ii,j)=V(ii)*V(j)*B(ii,j); 
                else
                    H(ii,ii)=V(ii)^2*B(ii,ii);
                end
            end
        end
    end
end
for ii=2:bus
    if node(ii,1)==2
        for j=2:bus
            if node(j,1)==2
                if ii~=j
                    L(ii,j)=V(ii)*V(j)*B(ii,j);
                else
                    L(ii,ii)=V(ii)^2*B(ii,ii);
                end
            end
        end
    end
end
M=zeros(bus,bus);
N=zeros(bus,bus);
Jacobi=-[H,M;N,L];
Jacobi(all(Jacobi==0,2),:)=[];%去掉全零行
Jacobi(:,all(Jacobi==0,1))=[];%去掉全零列


