#write first neural network

#network architechture
n_lay <- 4 #number of layers, with input and output layers included
n_inp <- 4 #number of input nueral units
n_out <- 4 #number of output nueral units
v_n_nueral <- 20 #number of nueral units in the first hidden layer
w_n_nueral <- 30 #number of nueral units in the second hidden layer
alpha <- 0.02 #learning rate
lambda <- 0.2 #decay weight

.

#sigmoid function
sigmoid <- function(x){
  return(1/(1+exp(-x)))
}

#tanh function
tanh <- function(x){
  return((exp(x)-exp(-x))/(exp(x)+exp(-x)))
}

#making data, in total 400 examples with 4 features
set.seed(11)
train <- matrix(runif(1600,max = 1,min = -1),nrow=400,ncol=4)

#parameters initialization

#bias in the input layer
set.seed(12)
bi <- matrix(rnorm(v_n_nueral),ncol = 1,nrow = v_n_nueral)
#bias in the 1st hidden layer
set.seed(13)
bv <- matrix(rnorm(w_n_nueral),ncol = 1,nrow = w_n_nueral)
#bias in the 2nd hidden layer
set.seed(14)
bw <- matrix(rnorm(n_out),ncol = 1,nrow = n_out)

#weights
wiv <- matrix(rnorm(n_inp,v_n_nueral),ncol = n_inp,nrow = v_n_nueral)
wvw <- matrix(rnorm(v_n_nueral,w_n_nueral),ncol = v_n_nueral,nrow = w_n_nueral)
wwo <- matrix(rnorm(w_n_nueral,n_out),ncol = w_n_nueral,nrow = n_out)

#training
tr_nn <- function(epoch=k){
  loss <- 0
  thelta_wiv <- matrix(0,ncol = n_inp,nrow = v_n_nueral)
  thelta_wvw <- matrix(0,ncol = v_n_nueral, nrow = w_n_nueral)
  thelta_wwo <- matrix(0,ncol = w_n_nueral, nrow = n_out)
  b_i <- matrix(0,ncol = 1,nrow = v_n_nueral)
  b_v <- matrix(0,ncol = 1,nrow = w_n_nueral)
  b_w <- matrix(0,ncol = 1,nrow = n_out)
  
  loss_df <- data.frame()
  
  for(j in 1:epoch){
    for(i in 1:dim(train)[1]){
      #forward propogattion
      z2 <- as.matrix(colSums(t(wiv) * train[i,1:4]),ncol=1)+bi
      a2 <- sigmoid(z2)
      z3 <- as.matrix(colSums(t(wvw) * as.numeric(a2)),ncol=1)+bv
      a3 <- tanh(z3)
      z4 <- as.matrix(colSums(t(wwo) * as.numeric(a3)),ncol=1)+bw
      a4 <- sigmoid(z4)
      loss <- loss + 0.5*sum((a4-train[i,1:4])^2)
      
      #backward propogation
      thelta_4 <- -(train[i,1:4]-a4) * a4*(1-a4)
      thelta_3 <- (t(wwo) %*% thelta_4) * (1-a3^2)
      thelta_2 <- (t(wvw) %*% thelta_3) * a2*(1-a2)
      
      d_thelta1 <- thelta_2 %*% t(train[i,1:4])
      d_thelta2 <- thelta_3 %*% t(a2)
      d_thelta3 <- thelta_4 %*% t(a3)
      thelta_wiv <- thelta_wiv + d_thelta1
      thelta_wvw <- thelta_wvw + d_thelta2
      thelta_wwo <- thelta_wwo + d_thelta3
      
      d_bi <- thelta_2
      d_bv <- thelta_3
      d_bw <- thelta_4
      b_i <- b_i + d_bi
      b_v <- b_v + d_bv
      b_w <- b_w + d_bw
    }
    loss <- 1/dim(train)[1] * loss +lambda/2*(sum(colSums(wiv^2))+sum(colSums(wvw^2))+sum(colSums(wwo^2)))
    wiv <- wiv - alpha*(1/(dim(train)[1])*(thelta_wiv) + lambda*(wiv))
    wvw <- wvw - alpha*(1/(dim(train)[1])*(thelta_wvw) + lambda*(wvw))
    wwo <- wwo - alpha*(1/(dim(train)[1])*(thelta_wwo) + lambda*(wwo))
    bi <- bi - alpha * (1/dim(train)[1])*(b_i)
    bv <- bv - alpha * (1/dim(train)[1])*(b_v)
    bw <- bw - alpha * (1/dim(train)[1])*(b_w)
    print(paste('epoch no.',j,':','        loss = ',loss,sep=''))
    loss_df <- rbind(loss_df,c(j,loss))
    flush.console()
    plot(x=loss_df[,1],y=loss_df[,2],type='l',ylim = c(0,max(loss_df[,2])),xlim = range(loss_df[,1]),
         xlab = '# iteration', ylab = 'loss',main = 'My four-layer nn',sub = paste(j,'th iteration',sep=''))
    Sys.sleep(0.09)
  }
return(list(bi=bi,bv=bv,bw=bw,wiv=wiv,wvw=wvw,wwo=wwo))
  
}


te_nn <- function(x,parameters=params){
  bi <- parameters$bi
  bv <- parameters$bv
  bw <- parameters$bw
  wiv <- parameters$wiv
  wvw <- parameters$wvw
  wwo <- parameters$wwo
  
  z2 <- as.matrix(colSums(t(wiv) * train[x,]),ncol=1)+bi
  a2 <- sigmoid(z2)
  z3 <- as.matrix(colSums(t(wvw) * as.numeric(a2)),ncol=1)+bv
  a3 <- tanh(z3)
  z4 <- as.matrix(colSums(t(wwo) * as.numeric(a3)),ncol=1)+bw
  a4 <- sigmoid(z4)
  return(data.frame(real=train[x,],pred=a4))
}

params <- tr_nn(2000)
lapply(1:10,te_nn)
