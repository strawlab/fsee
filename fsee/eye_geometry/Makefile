# This (GNU) Makefile is used to maintain the data products that fsee
# makes. It is useful to keep the directories clean but up-to-date.
#
# Run with:
#  make

default: eye_geometry_data_files

synthetic_products = precomputed_synthetic.py receptor_weight_matrix_64_synthetic.mat receptors_by_phi_synthetic.mat precomputed_synthetic.m
buchner71_products = precomputed_buchner71.py receptor_weight_matrix_64_buchner71.mat

clean:
	rm -f $(synthetic_products)
	rm -f $(buchner71_products)

eye_geometry_data_files: synthetic buchner71

synthetic: $(synthetic_products)

buchner71: $(buchner71_products)

$(synthetic_products): precompute_synthetic_optics.py
	python precompute_synthetic_optics.py

$(buchner71_products): precompute_buchner71_optics.py
	python precompute_buchner71_optics.py

.PHONY: clean eye_geometry_data_files default synthetic buchner71
