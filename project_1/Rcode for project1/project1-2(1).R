#import data
d <- read.table(file='file:///Users/ran/Desktop/project1/data/prostate.data.txt', head=TRUE)

d1 <- d[sample.int(97),] 
d1 <- d
xmean <- matrix(data = NA, ncol = 9)
xvar <- matrix(data = NA, ncol = 9)

#initialization of xmean, xvar, x, y, xtest, ytest
for (j in 1:8){
  xmean[j] <- mean(d1[1:50, j])
  xvar[j] <- sqrt(sum((d1[1:50, j]-xmean[j])^2)/50)

  for (i in 1:97)
    d1[i,j] <- (d1[i,j]-xmean[j])/xvar[j]
}
ymean <- mean(d1[1:50,9])
for (i in 1:97)
  d1[i,9] <- d1[i,9]-ymean
  
x <- as.matrix(d1[1:50, 1:8])
y <- as.matrix(d1[1:50, 9])
xtest <- as.matrix(d1[51:97, 1:8])
ytest <- as.matrix(d1[51:97, 9])

#function ridge, get theta for different x,y,d2
ridge <- function(x,y,d2){
  theta <- ginv(t(x)%*%x+diag(d2,8),tol=0) %*% t(x) %*% y
  theta
}

#initialization of TrainError, TestError, d2 and th
th <- matrix(data = NA, nrow = 1000, ncol = 8)
TrainError <- matrix(data = NA, ncol = 1000)
TestError <- matrix(data = NA, ncol = 1000)
d2 <- 10^((1:1000)*0.006-2)

# for 1000 different d2, get theta, TrainError and TestError
for (i in 1:1000){
  th[i, ] <- ridge(x,y,d2[i])
  yhat <- x%*%th[i, ]
  testyhat <- xtest%*%th[i, ]
 
  TrainError[i] <- sqrt(sum((y-yhat)^2)/sum((y+ymean)^2))
  TestError[i] <- sqrt(sum((ytest-testyhat)^2)/sum((ytest+ymean)^2))
}

par(mfrow=c(1,1))
#draw the plots of d2-theta
plot(d2, th[,1], xlim = c(0.01,10000), ylim = c(-0.2,0.6),log='x', type='l',col=1,
     ylab=expression(theta), xlab = expression(delta^2), lwd=2)
for (i in 2:8)
  lines(d2, th[,i],col=i,lwd=2)

legend.text <- colnames(d1)[-9]
legend('topright', legend = legend.text,col=c(1:8),lty=1,cex=0.6)
grid(lty=1)

#draw polts of d2-TrainError,TestError
plot(d2, TrainError, ylim= c(0.25,0.55), log='x', type='l',lwd=2,
     xlab=expression(delta^2),ylab='error')
lines(d2,TestError,lwd=2,col=2)
legend('bottomright',legend=c('Train','Test'),col=c(1,2),lty=1,cex=1)
grid(lty=1)
