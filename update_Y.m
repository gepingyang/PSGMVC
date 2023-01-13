function [Y, obj] = update_Y(P, Y)

    YP = sum((Y'*P).^2, 2); 
    YTY = sum(Y)';

    m_all = vec2ind(Y')';

    obj(1) = sum(YP ./ YTY);
    for iter = 2:30
        for i = 1:size(Y, 1)
            m = m_all(i);
            if YTY(m) == 1 
                % avoid generating empty cluster
                continue;
            end

            Y_A = Y'*P*P(i,:)';
            YP_i = YP + 2 * Y_A + sum(P(i,:).^2);
            YP_i(m) = YP(m);
            YTY_i = YTY + 1;
            YTY_i(m) = YTY(m);  
            YP_0 = YP;
            YP_0(m) = YP(m) - 2 * Y_A(m) + sum(P(i,:).^2);
            YTY_0 = YTY;
            YTY_0(m) = YTY(m) - 1;
            T = YP_i ./ YTY_i - YP_0 ./ YTY_0 ;
            [~, p] = max(T);
            if p ~= m
                YP([m, p]) = [YP_0(m), YP_i(p)];
                YTY([m, p]) = [YTY_0(m), YTY_i(p)];
                Y(i, [p, m]) = [1, 0];
                m_all(i) = p;
            end
        end
        obj(iter) = sum(YP ./ YTY);
        if iter > 2 && (obj(iter) - obj(iter - 1)) / obj(iter - 1) < 1e-4
            break;
        end
    end
end