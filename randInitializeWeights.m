function W = randInitializeWeights(L_in, L_out)
  
W = zeros(L_out, 1 + L_in);
epsInit = 0.12;
W = rand(L_out,L_in+1)*(2*epsInit) - epsInit;

end
