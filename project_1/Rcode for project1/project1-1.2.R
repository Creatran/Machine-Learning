#install.packages("MASS")
library(MASS)

basicdata <- read.table(file='file:///Users/ran/Desktop/project1/data/basicData.txt', header=T)
x <- basicdata$X
y <- basicdata$y
xtest <- basicdata$Xtest
ytest <- basicdata$Ytest

d <- dim(basicdata)

makeXpoly <- function(x,deg){
  m <- matrix(data=NA,ncol=deg+1,nrow=d)
  for (i in (0 : deg)){
    m[,i+1] <- x^i
  } 
  m
}
leastSquaresBasis <- function(x,y,deg){
  
  Xpoly <- makeXpoly(x,deg)
  b <<- ginv(t(Xpoly)%*%Xpoly,tol=0) %*%t(Xpoly)%*%y
  
  Xpoly%*%b
}

#tt <- matrix( nrow = 11, ncol=2)
par(mfrow=c(4,3))
for (i in 0:10){
  yhat <- leastSquaresBasis(x,y,i)
  plot(x,yhat, type = 'l',xlim=c(-10,10),ylim = c(-300,400),
       col='red', main = c("training data",'deg=',i),lwd=5)
  points(x,y,pch =20,col='blue',cex=0.3)
  
  TrainError <- sum((y-yhat)^2)/d[1]
  ytesthat <- makeXpoly(xtest,i) %*% b
  TestError <- sum((ytest-ytesthat)^2)/d[1]
  
#  tt[i+1, ] <- c(TrainError,TestError)
  print(c('k=',i))
  print(c(' TrainError ', TrainError, ' TestError ', TestError))
}
#print(tt)
#write.table(file="file:///Users/ran/Desktop/ee.xlsx", tt)

