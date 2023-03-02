function [ Jacobi ] = Jacobi_NR(bus,B2,V,theta,B,G,P,Q)
for ii=1:bus
    if B2(ii,1)~=1
        for j=1:bus
            if B2(j,1)~=1
                if ii~=j
                   H(ii,j)=-V(ii)*V(j)*(G(ii,j)*sin(theta(ii)-theta(j))-B(ii,j)*cos(theta(ii)-theta(j))); 
                else
                    H(ii,ii)=V(ii)^2*B(ii,ii)+Q(ii);
                end
            end
        end
    end
end

for ii=2:bus
    for j=1:bus
        if B2(j,1)==2
            if ii~=j
                N(ii,j)=-V(ii)*V(j)*(G(ii,j)*cos(theta(ii)-theta(j))+B(ii,j)*sin(theta(ii)-theta(j)));
            else
                N(ii,ii)=-V(ii)^2*G(ii,ii)-P(ii);
            end
        end
    end
end
for ii=2:bus
    if B2(ii,1)==2    
        for j=2:bus
            if ii~=j
                K(ii,j)=V(ii)*V(j)*(G(ii,j)*cos(theta(ii)-theta(j))+B(ii,j)*sin(theta(ii)-theta(j)));
            else
                K(ii,j)=V(ii)^2*G(ii,ii)-P(ii);
            end
        end
    end
end
for ii=2:bus
    if B2(ii,1)==2
        for j=2:bus
            if B2(j,1)==2
                if ii~=j
                    L(ii,j)=-V(ii)*V(j)*(G(ii,j)*sin(theta(ii)-theta(j))-B(ii,j)*cos(theta(ii)-theta(j)));
                else
                    L(ii,ii)=V(ii)^2*B(ii,ii)-Q(ii);
                end
            end
        end
    end
end
Jacobi=-[H,N;K,L];
Jacobi(all(Jacobi==0,2),:)=[];%去掉全零行
Jacobi(:,all(Jacobi==0,1))=[];%去掉全零列
end

