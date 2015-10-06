#Makefile template by Guojie Luo

#ROOT_CCFLAGS := $(ROOT_CCFLAGS)

MSbeam_OUT = $(ROOT_NAME)
MSbeam_LIB =
MSbeam_SRC = main.c \
                        utility.c \
						proj.c \

all :: $(ROOT_BIN)/$(MSbeam_OUT)

MSbeam_LDFLAGS = $(MSbeam_LIB) $(ROOT_LDFLAGS)
MSbeam_OBJ = $(MSbeam_SRC:%.c=$(ROOT_BUILD)/%.o)
MSbeam_DEP = $(patsubst %,%.d,$(basename $(MSbeam_OBJ)))

$(MSbeam_OUT) :: $(MSbeam_OBJ)
	$(ROOT_LNK) -o $(ROOT_BIN)/$(MSbeam_OUT) $^ $(MSbeam_LDFLAGS)

$(ROOT_BIN)/$(MSbeam_OUT) :: $(MSbeam_OBJ)
	$(ROOT_LNK) -o $@ $^ $(MSbeam_LDFLAGS)

clean ::
	rm -f $(MSbeam_DEP)
	rm -f $(MSbeam_OBJ)
	rm -f $(ROOT_BIN)/$(MSbeam_OUT)

-include $(MSbeam_DEP)
