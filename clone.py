__author__ = "Sam Prestwood"
__email__ = "swp2sf@virginia.edu"

"""
clone.py
Gradient Domain Cloning
Computational Photography, Fall 2014
"""

# imports:

import numpy
import matplotlib.pyplot as plt
import skimage.io
import scipy.sparse
import scipy.sparse.linalg

# globals:

BACKGROUND_IMG = "imgs/Mona_Lisa.jpg"
FOREGROUND_IMG = "imgs/ron2.jpg"
MATTE_IMG = "imgs/ron_matte.png"

# functions:

def same_dim(b_img, f_img, m_img):
	"""confirms that the background, foreground, and matte images all have the\
	same dimensions"""

	if b_img.shape != f_img.shape or f_img.shape != m_img.shape or \
	m_img.shape != b_img.shape:
		print "Error: image shapes don't match"
		print "background image:", b_img.shape
		print "foreground image:", f_img.shape
		print "matte image:", m_img.shape
		return False

	else:
		return True

def naive_clone():
	"""Simply pastes the foreground on top of the background"""

	global BACKGROUND_IMG, FOREGROUND_IMG, MATTE_IMG

	b_img = skimage.img_as_float(skimage.io.imread(BACKGROUND_IMG))
	f_img = skimage.img_as_float(skimage.io.imread(FOREGROUND_IMG))
	m_img = skimage.img_as_float(skimage.io.imread(MATTE_IMG))
	o_img = numpy.zeros((m_img.shape[0], m_img.shape[1], 3))

	# make sure all the images have the same dimensions
	if same_dim(b_img, f_img, m_img):
		for r in range(m_img.shape[0]):
			for c in range(m_img.shape[1]):
				#print r, c
				if numpy.allclose(m_img[r, c], [0, 0, 0]): 
					o_img[r, c] = b_img[r, c]
				else:
					o_img[r, c] = f_img[r, c]
			print "creating row", r, "of", m_img.shape[0]

		plt.imshow(o_img)
		plt.show()

def naive_clone_fast():
	"""same as naive_clone(), but uses built-in numpy operations to speed \
	things up"""

	global BACKGROUND_IMG, FOREGROUND_IMG, MATTE_IMG

	b_img = skimage.img_as_float(skimage.io.imread(BACKGROUND_IMG))
	f_img = skimage.img_as_float(skimage.io.imread(FOREGROUND_IMG))
	m_img = skimage.img_as_float(skimage.io.imread(MATTE_IMG))
	
	if same_dim(b_img, f_img, m_img):
		o_img = m_img * (f_img - b_img) + b_img
		plt.imshow(o_img)
		plt.show()

def create_matte_mapping(m_img):
	"""given the matte image, returns a 1D array of (x, y) coordinates that map\
	a pixel to its coordinates in the matte image"""

	foreward_map = {}
	backward_map = {}
	d_omega_f = {}
	d_omega_b = {}

	i_map = 0
	i_omega = 0
	for r in range(m_img.shape[0]):
		for c in range(m_img.shape[1]):

			# create matte mappings:
			
			if numpy.linalg.norm(m_img[r, c]) > 0.8:
				foreward_map[i_map] = (r, c)
				backward_map[(r, c)] = i_map
				i_map += 1

			# create boundary region mappings:
			else:
			 	# up:
				if r > 0 and numpy.linalg.norm(m_img[r - 1, c]) > 0.8:
					d_omega_f[i_omega] = (r, c)
					d_omega_b[(r, c)] = i_omega
					i_omega += 1

				# right:
				if c < m_img.shape[1] - 1 and numpy.linalg.norm(m_img[r, \
				c + 1]) > 0.8:
					d_omega_f[i_omega] = (r, c)
					d_omega_b[(r, c)] = i_omega
					i_omega += 1

				# down:
				if r < m_img.shape[0] - 1 and numpy.linalg.norm(m_img[r + 1, \
				c]) > 0.8:
					d_omega_f[i_omega] = (r, c)
					d_omega_b[(r, c)] = i_omega
					i_omega += 1

				# left
				if c > 0 and numpy.linalg.norm(m_img[r, c - 1]) > 0.8:
					d_omega_f[i_omega] = (r, c)
					d_omega_b[(r, c)] = i_omega
					i_omega += 1

	return (foreward_map, backward_map, d_omega_f, d_omega_b)

def gradient_clone():
	"""takes the gradient of the background and foreground images, then pastes\
	them together"""

	global BACKGROUND_IMG, FOREGROUND_IMG, MATTE_IMG

	b_img = skimage.img_as_float(skimage.io.imread(BACKGROUND_IMG))
	f_img = skimage.img_as_float(skimage.io.imread(FOREGROUND_IMG))
	m_img = skimage.img_as_float(skimage.io.imread(MATTE_IMG))

	if same_dim(b_img, f_img, m_img):
		b_img = numpy.gradient(b_img)[0]
		b_img = numpy.clip(b_img, 0.0, 1.0)

		f_img = numpy.gradient(f_img)[0]
		f_img = numpy.clip(f_img, 0.0, 1.0)

		o_img = m_img * (f_img - b_img) + b_img

		plt.imshow(o_img)
		plt.show()

def gradient_clone_poisson():
	"""attempts to take the gradient_clone() one step further by 'solving' the\
	implied poisson equation"""
	
	global BACKGROUND_IMG, FOREGROUND_IMG, MATTE_IMG

	b_img = skimage.img_as_float(skimage.io.imread(BACKGROUND_IMG))
	f_img = skimage.img_as_float(skimage.io.imread(FOREGROUND_IMG))
	m_img = skimage.img_as_float(skimage.io.imread(MATTE_IMG))

	if same_dim(b_img, f_img, m_img):

		# create mappings:

		print "creating matte pixel mapping..."

		mapping = create_matte_mapping(m_img)
		f_matte_map = mapping[0]
		b_matte_map = mapping[1]
		f_boundary_map = mapping[2]
		b_boundary_map = mapping[3]

		print "done"

		# create x vectors and A matrices:
		
		print "creating b vectors and A matrices..."

		b_r = numpy.zeros(len(f_matte_map))
		b_g = numpy.zeros(len(f_matte_map))
		b_b = numpy.zeros(len(f_matte_map))

		A_r = scipy.sparse.lil_matrix((len(f_matte_map), len(f_matte_map)))
		A_g = scipy.sparse.lil_matrix((len(f_matte_map), len(f_matte_map)))
		A_b = scipy.sparse.lil_matrix((len(f_matte_map), len(f_matte_map)))

		print "done"

		# fill vectors and matrices:

		print "filling vectors and matrices..."

		for pixel in f_matte_map:
			r = f_matte_map[pixel][0]
			c = f_matte_map[pixel][1]

			# calculate n_p:

			n_p = 0

			if r > 0: # up
				n_p += 1
			if c < m_img.shape[1] - 1: # right:
				n_p += 1
			if r < m_img.shape[0] - 1: # down:
				n_p += 1
			if c > 0: # left
				n_p += 1

			# populate A matrices:

			A_r[b_matte_map[(r, c)], b_matte_map[(r, c)]] = n_p
			A_g[b_matte_map[(r, c)], b_matte_map[(r, c)]] = n_p
			A_b[b_matte_map[(r, c)], b_matte_map[(r, c)]] = n_p

			if (r - 1, c) in b_matte_map: # up
				A_r[b_matte_map[(r, c)], b_matte_map[(r - 1, c)]] = -1
				A_g[b_matte_map[(r, c)], b_matte_map[(r - 1, c)]] = -1
				A_b[b_matte_map[(r, c)], b_matte_map[(r - 1, c)]] = -1
			if (r, c + 1) in b_matte_map: # right
				A_r[b_matte_map[(r, c)], b_matte_map[(r, c + 1)]] = -1
				A_g[b_matte_map[(r, c)], b_matte_map[(r, c + 1)]] = -1
				A_b[b_matte_map[(r, c)], b_matte_map[(r, c + 1)]] = -1
			if (r + 1, c) in b_matte_map: # down
				A_r[b_matte_map[(r, c)], b_matte_map[(r + 1, c)]] = -1
				A_g[b_matte_map[(r, c)], b_matte_map[(r + 1, c)]] = -1
				A_b[b_matte_map[(r, c)], b_matte_map[(r + 1, c)]] = -1
			if (r, c - 1) in b_matte_map: #left
				A_r[b_matte_map[(r, c)], b_matte_map[(r, c - 1)]] = -1
				A_g[b_matte_map[(r, c)], b_matte_map[(r, c - 1)]] = -1
				A_b[b_matte_map[(r, c)], b_matte_map[(r, c - 1)]] = -1

			# populate b vectors:

			if (r - 1, c) in b_boundary_map: # up
				b_r[b_matte_map[(r, c)]] += b_img[r - 1, c, 0]
				b_g[b_matte_map[(r, c)]] += b_img[r - 1, c, 1]
				b_b[b_matte_map[(r, c)]] += b_img[r - 1, c, 2]
			if (r, c + 1) in b_boundary_map: # right
				b_r[b_matte_map[(r, c)]] += b_img[r, c + 1, 0]
				b_g[b_matte_map[(r, c)]] += b_img[r, c + 1, 1]
				b_b[b_matte_map[(r, c)]] += b_img[r, c + 1, 2]
			if (r + 1, c) in b_boundary_map: # down
				b_r[b_matte_map[(r, c)]] += b_img[r + 1, c, 0]
				b_g[b_matte_map[(r, c)]] += b_img[r + 1, c, 1]
				b_b[b_matte_map[(r, c)]] += b_img[r + 1, c, 2]
			if (r, c - 1) in b_boundary_map: #left
				b_r[b_matte_map[(r, c)]] += b_img[r, c - 1, 0]
				b_g[b_matte_map[(r, c)]] += b_img[r, c - 1, 1]
				b_b[b_matte_map[(r, c)]] += b_img[r, c - 1, 2]

			if r > 0: # up
				b_r[b_matte_map[(r, c)]] += (f_img[r, c, 0] - f_img[r - 1, c, 0])
				b_g[b_matte_map[(r, c)]] += (f_img[r, c, 1] - f_img[r - 1, c, 1])
				b_b[b_matte_map[(r, c)]] += (f_img[r, c, 2] - f_img[r - 1, c, 2])
			if c < m_img.shape[1] - 1: # right:
				b_r[b_matte_map[(r, c)]] += (f_img[r, c, 0] - f_img[r, c + 1, 0])
				b_g[b_matte_map[(r, c)]] += (f_img[r, c, 1] - f_img[r, c + 1, 1])
				b_b[b_matte_map[(r, c)]] += (f_img[r, c, 2] - f_img[r, c + 1, 2])
			if r < m_img.shape[0] - 1: # down:
				b_r[b_matte_map[(r, c)]] += (f_img[r, c, 0] - f_img[r + 1, c, 0])
				b_g[b_matte_map[(r, c)]] += (f_img[r, c, 1] - f_img[r + 1, c, 1])
				b_b[b_matte_map[(r, c)]] += (f_img[r, c, 2] - f_img[r + 1, c, 2])
			if c > 0: # left
				b_r[b_matte_map[(r, c)]] += (f_img[r, c, 0] - f_img[r, c - 1, 0])
				b_g[b_matte_map[(r, c)]] += (f_img[r, c, 1] - f_img[r, c - 1, 1])
				b_b[b_matte_map[(r, c)]] += (f_img[r, c, 2] - f_img[r, c - 1, 2])

		print "done"

		# calculate solution to poisson equation

		print "calculating solution to Poisson equation..."

		u_r = numpy.clip(scipy.sparse.linalg.cg(A_r, b_r)[0], 0.0, 1.0)
		u_g = numpy.clip(scipy.sparse.linalg.cg(A_g, b_g)[0], 0.0, 1.0)
		u_b = numpy.clip(scipy.sparse.linalg.cg(A_b, b_b)[0], 0.0, 1.0)

		print "done"

		# write pixel values to image

		print "writing pixel values to image"

		one = numpy.ones((m_img.shape[0], m_img.shape[1], 3))
		o_img = (one - m_img) * b_img

		for pixel in f_matte_map:
			r = f_matte_map[pixel][0]
			c = f_matte_map[pixel][1]

			#print r, c, u_r, u_r[pixel]

			o_img[r, c, 0] = u_r[pixel]
			o_img[r, c, 1] = u_g[pixel]
			o_img[r, c, 2] = u_b[pixel]

		print "done"

		plt.imshow(o_img)
		plt.show()

# main:

if __name__ == "__main__":
	#naive_clone()
	#naive_clone_fast()
	#gradient_clone()
	gradient_clone_poisson()