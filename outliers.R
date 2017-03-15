log <- read_delim("~/SDC/p3/p4/log.csv",
";", escape_double = FALSE, col_names = FALSE,
trim_ws = TRUE, skip = 1)

m <- ggplot(log) + geom_density(aes(x = X3)) + geom_vline(xintercept = c(3*madx3, -3*madx3), colour = "red")
m + geom_text((aes(x = 3*madx3, y = 0.00005,label = "MAD * 3")), colour = "red", vjust = 1, angle = 90) +
  xlab("Right lane distance") + 
  ggtitle("Right lane distance distribution")


m <- ggplot(log) + geom_density(aes(x = X2)) + geom_vline(xintercept = c(3*madx2, -3*madx2), colour = "red")
m + geom_text((aes(x = 3*madx2, y = 0.00005,label = "MAD * 3")), colour = "red", vjust = 1, angle = 90) + 
  xlab("Left lane distance") + 
  ggtitle("Left lane distance distribution")



