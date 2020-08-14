library(MASS)
plot(mcycle)
motor_matrix = as.matrix(mcycle)
system.time({y = locpoly(motor_matrix, 0.00, 10)})
y = cbind(motor_matrix[,'times'],y)
plot(motor_matrix)
lines(y,col = 'red')


n <- 1000 # number of data points
x1 <- runif(n,0,0.9)
x2 <- runif(n,0,0.95)
x3 <- runif(n,0,0.8)
b <- 2
c.unif <- runif(n)
c.norm <- rnorm(n)
amp <- 2

# generate data and calculate "y"
set.seed(1)
y1 <- a*sin(b*t)+c.unif*amp # uniform error
y2 <- a*sin(x1)+exp(x2)+x3^2+c.norm*amp # Gaussian/normal error
y2 = cbind(x1,x2,x3,y2)
y2 = as.matrix(y2)
plot(y2)
system.time({wlocpoly = locpoly(y2)})
z = y2[,'y2'] -wlocpoly
z


n <- 100 # number of data points
x1 <- runif(n,0,0.9)
x2 <- runif(n,0,0.95)
x3 <- runif(n,0,0.8)
b <- 2
c.unif <- runif(n)
c.norm <- rnorm(n)
amp <- 2

# generate data and calculate "y"
set.seed(1)
y1 <- a*sin(b*t)+c.unif*amp # uniform error
y2 <- a*sin(x1)+exp(x2)+x3^2+c.norm # Gaussian/normal error
y2 = cbind(x1,x2,x3,y2)
y2 = as.matrix(y2)
plot(y2)
system.time({wlocpoly = locpoly(y2)})
z = y2[,'y2'] -wlocpoly

n = 500
x_try = seq(0, pi*10, length.out =n)
y_try = sin(x_try) 
matrix_try = as.matrix(cbind(x_try,y_try))
plot(matrix_try)
wloc = locpoly(matrix_try)
wloc = cbind
plot(wloc)
#install.packages("locfit")
#library(locfit)
#z2 = locfit(y2~x1+x2+x3)
#system.time({loess1 <- loess(y2~x1+x2+x3)})

