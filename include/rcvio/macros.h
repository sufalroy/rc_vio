#pragma once

#include <boost/shared_ptr.hpp>

#define POINTER_TYPEDEFS(TypeName)                      \
    typedef boost::shared_ptr<TypeName> Ptr;            \
    typedef boost::shared_ptr<const TypeName> ConstPtr; \
    void definePointerTypedefs##__FILE__##__LINE__(void)
