d <- read.table(file='file:///Users/ran/Desktop/project1/data/prostate.data.txt', head=TRUE)

#initialization of k d1 d0 st(stands for step in each training data）

k=3
d0 <- floor(97/k)
st <- matrix(1)
for (i in 2:k)
  st[i] <- st[i-1]+d0
st[k+1] <- 98

#function ridge get theta for different x,y,d2
ridge <- function(x,y,d2){
  theta <- ginv(t(x)%*%x+diag(d2,8),tol=0) %*% t(x) %*% y
  theta
}

#initialization of TrainError, TestError and th(stands for different theta)
TrainError <- matrix(data = 0, ncol = 1000)
TestError <- matrix(data = 0, ncol = 1000)
th <- matrix(data = NA, nrow = 1000, ncol = 8)
d2 <- 10^((1:1000)*0.006-2)

# repeat to get k different TrainError and TestError
for (num in 1:k){
  set.seed(123)
  d1 <- d[sample.int(97),]
  #initialization of different x, y, xtest, ytest
  for (j in 1:8){
    xmean[j] <- mean(d1[-st[num]:-(st[num+1]-1), j])
    xvar[j] <- sqrt(sum((d1[-st[num]:-(st[num+1]-1), j]-xmean[j])^2)/(97-(st[num+1]-st[num])))
    
    for (i in 1:97)
      d1[i,j] <- (d1[i,j]-xmean[j])/xvar[j]
  }
  ymean <- mean(d1[-st[num]:-(st[num+1]-1), 9])
  for (i in 1:97)
    d1[i,j] <- d1[i,j]-ymean

  x <- as.matrix(d1[-st[num]:-(st[num+1]-1), 1:8])
  y <- as.matrix(d1[-st[num]:-(st[num+1]-1), 9])
  xtest <- as.matrix(d1[st[num]:(st[num+1]-1), 1:8])
  ytest <- as.matrix(d1[st[num]:(st[num+1]-1), 9])
  
  # for 1000 different d2, get theta, TrainError and TestError
  for (i in 1:1000){
    th[i, ] <- ridge(x, y, d2[i])
    yhat <- x %*% th[i, ]
    testyhat <- xtest%*%th[i, ]
   
    TrainError[i] <- TrainError[i]+sqrt(sum((y-yhat)^2)/sum((y+ymean)^2))
    TestError[i] <- TestError[i]+sqrt(sum((ytest-testyhat)^2)/sum((ytest+ymean)^2))
  }
}

#draw plots of d2-TrainError,TestError
TrainError <- TrainError/k
TestError <- TestError/k

plot(d2, TrainError, log='x', type='l',col=1,ylab='Error')
lines(d2,TestError,col=2)
legend('bottomright',legend=c('Train','Test'),col=c(1,2),lty=1,cex=1)
grid()

