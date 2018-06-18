import os 
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

tf.app.flags.DEFINE_string('label','Test','This is the name of the test')

FLAGS = tf.app.flags.FLAGS

dirname = os.path.realpath('.')
folderName = FLAGS.label
print (folderName)
newdir = dirname + '\\ModelsAndResults\\' + folderName

plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])
plt.xlabel('Months')
plt.ylabel('Books Read')
plt.savefig(newdir + '\\graph.png')
plt.show()

if not (os.path.exists(newdir)):
	os.makedirs(newdir)

f= open(newdir + '\\Results.txt',"w+")
f.write("This is the new line " + dirname + '\n')
f.write("This is the new line")
period = .9939999
validation_log_loss = 124124.343242
f.write(" period %02d: %0.2f" % (period, validation_log_loss))
