package mnist

import java.io.File
import java.io.FileInputStream

class MNISTData(input: File, labelsInput: File) {
  var stream = new FileInputStream(input)
  def read(len: Int) = {
    val arr = new Array[Byte](len)
    var ix = 0
    while ({
      val read = stream.read(arr, ix, len - ix)
      ix += read
      ix < len
    }) {}
    arr.toVector.map(_.toInt & 0xFF)
  }
  def readNum() = {
    val Vector(a, b, c, d) = read(4)
    (a << 24) | (b << 16) | (c << 8) | d
  }
  val magic = readNum()
  if (magic != 0x00000803)
    throw new Exception()
  val nImgs = readNum()
  val rows = readNum()
  val cols = readNum()
  val imgs = Vector.tabulate(nImgs) { n =>
    if (n % 2400 == 0) {
      println(s"Loading Images: ${(n.toDouble / nImgs * 100).toInt}%")
    }
    Vector.fill(rows) {
      read(cols).map(_.toDouble / 255)
    }.transpose
  }
  stream.close()
  stream = new FileInputStream(labelsInput)
  val labelMagic = readNum()
  if (labelMagic != 0x00000801)
    throw new Exception()
  if (readNum() != nImgs)
    throw new Exception()
  val labels = read(nImgs)
  stream.close()
}