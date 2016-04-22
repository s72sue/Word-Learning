library(RcppCNPy)
getwd()

#z <- npyLoad("Downloads/test_symm.npy")


#z <- npyLoad("Downloads/result_files/random.npy")
#a <- npyLoad("Downloads/result_files/random_acc.npy")

z <- npyLoad("Downloads/result_files/eqlikely.npy")
a <- npyLoad("Downloads/result_files/eqlikely_acc.npy")

#z <- npyLoad("Downloads/normal_small4.npy")
#a <- npyLoad("Downloads/normal_small_acc4.npy")

#z <- npyLoad("Downloads/result_files/normal.npy")
#a <- npyLoad("Downloads/result_files/normal_acc.npy")

# Acceptance Probability
mean(a)

### cumulative acceptance prob. over time
plot(cumsum(a)/(1:length(a)),type='l')

# original histogram, traceplot and autocorrelation plots of complete data
# histogram
#hist(z, prob=TRUE, main="Histogram of the posterior", breaks=seq(-1,6,1), cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
hist(z, prob=TRUE, main="Histogram of the posterior", breaks=seq(0,37,1), cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
# traceplot
plot(z, type='l', main="traceplot", cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
# autocorrelation plot
acf(z, cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)


z[1:100]

# remove first 10,000 samples
bq2 = z[-c(1:10^4)]
bq2[1:100]
length(bq2)

# lag of 50
z_lag = bq2[seq(1,4*10^4, by=50)] 
length(z_lag)

# histogram, traceplot and autocorrelation plots after processing
hist(z_lag, prob=TRUE, main="Histogram of the posterior", breaks=seq(0,37,1))
plot(z_lag, type='l', main="traceplot after lag", cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
acf(z_lag, cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)


mean(z_lag)
sd(z_lag)


### 95% credible interval for lambda is 
quantile(z_lag, c(0.250, 0.975)) 



