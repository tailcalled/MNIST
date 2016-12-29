package mnist

import java.io.File
import nn.NeuralNetwork
import java.util.Random
import java.awt.image.BufferedImage
import javax.imageio.ImageIO

object MNISTClassifier {
  def main(args: Array[String]): Unit = {
    val data = new MNISTData(new File("train-images.idx3-ubyte"), new File("train-labels.idx1-ubyte"))
    val net = NeuralNetwork.make()
    val input = Vector.fill(27, 27)(net.input(0.0))
    val rand = new Random(0)
    val metarate = net.input(-0.00001)
    val ratecell = net.input(-0.001)
    val rategrad = net.input(0.0)
    val rate = net.sum(ratecell, net.product(rategrad, metarate))
    var weights = Vector[(net.Cell, net.Cell)]()
    def makeWeight(): net.Node = {
      val w = net.input(rand.nextGaussian())
      val g = net.input(0.0)
      weights :+= ((w, g))
      net.sum(w, net.product(g, rate))
    }
    def neuron(ins: Vector[net.Node])(implicit weight: () => net.Node) = {
      val bias = weight()
      val prod = ins.map(n => net.product(weight(), n))
      net.sigmoid(net.sum((bias +: prod):_*))
    }
    def layer(ins: Vector[net.Node], w: Int)(implicit weight: () => net.Node) = {
      Vector.fill(w)(neuron(ins))
    }
    def convLayer(sz: Int, prev: Vector[Vector[Vector[net.Node]]]) = {
      val source = Stream.continually(makeWeight())
      def unit(x: Int, y: Int) = {
        val sourceIt = source.iterator
        implicit val weight = () => sourceIt.next()
        val l = prev.drop(sz * x).take(sz).map(_.drop(sz * y).take(sz)).flatten.flatten
        layer(l, 4)
      }
      val s = prev.length / sz // assuming square image
      Vector.tabulate(s, s)(unit _)
    }
    println(data.nImgs)
    var l: Vector[Vector[Vector[net.Node]]] = input.map(_.map(Vector(_)))
    l = convLayer(3, l)
    l = convLayer(3, l)
    val out0 = layer(l.flatten.flatten, 10)(makeWeight _)
    val outNorm = net.sum(out0: _*)
    val out = out0.map(n => net.divide(n, outNorm))
    val expected = Vector.fill(10)(net.input(0.0))
    val diffs = Vector.tabulate(10)(i => 
      net.sum(out(i), net.product(net.constant(-1.0), expected(i)))
    )
    val errs = diffs.map(n => net.product(n, n))
    val err = net.sum(errs: _*)
    val batchSize = 10
    var avg = 0.0
    for (q <- 0 until 20) {
      for (i <- 0 until data.nImgs/batchSize) {
        var batch = (node: net.Node) => 0.0
        for (c <- 0 until batchSize) {
          val ix = i*batchSize + c
          val img = data.imgs(ix)
          for (x <- 0 until 27; y <- 0 until 27) {
            net(input(x)(y)) = img(x)(y)
          }
          for (e <- expected) net(e) = 0.0
          net(expected(data.labels(ix))) = 1.0
          net.reset()
          avg = avg * 0.995 + net(err) * 0.005
          val batch_ = batch
          val b = net.gradient(err)
          batch = (node) => batch_(node) + b(node)
        }
        for ((w, g) <- weights) {
          val g_ = net(g)
          net(g) = batch(w)
          net(w) = net(w) + net(rate) * g_
        }
        val rateg = net(rategrad)
        net(rategrad) = batch(rate)
        net(ratecell) = net(ratecell) + net(metarate) * rateg
        net(metarate) = net(metarate) - 0.00000001 * batch(metarate)
        if (i % 500 == 0) println(s"$avg ${net(rate)} ${net(metarate)}")
      }
    }
    for ((img, q) <- data.imgs.zipWithIndex) {
      val buf = new BufferedImage(27, 27, BufferedImage.TYPE_INT_ARGB)
      for (x <- 0 until 27; y <- 0 until 27) {
        def ch(c: Double) = (c*255).toInt max 0 min 255
        def col(c: Double) = (0xFF << 24) | (ch(c) << 16) | (ch(c) << 8) | ch(c)
        net(input(x)(y)) = img(x)(y)
        buf.setRGB(x, y, col(img(x)(y)))
      }
      net.reset()
      val ix = out.zipWithIndex.maxBy(n => net(n._1))._2
      new File(s"$ix").mkdir()
      ImageIO.write(buf, "png", new File(s"$ix/$q.png"))
    }
    ()
  }
}