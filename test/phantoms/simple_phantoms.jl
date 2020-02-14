function disc_phantom(img, a, b, r)
    height, width = size(img)
    #find the possibility which has most surrounding pixels
    disc = zeros(Float64, (height,width))
    for x = 1:width
        for y = 1:height
            disc[y,x] = (x-a)^2 + (y-b)^2 < r^2 ? 1.0 : 0.0
        end
    end
    return disc
end
