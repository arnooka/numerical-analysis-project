% altitude
zero = min(dji_sdk_gps_position.altitude);
altitude = dji_sdk_gps_position.altitude - zero;
altitude = table(altitude);
altitude_total = [altitude_total; altitude];

% timestep
min_time = min(dji_sdk_gps_position.header_times);
%t = round(dji_sdk_gps_position.header_times - min_time, 3);
t = round(dji_sdk_gps_position.header_times - min_time + ( table2array(t_total(end,:)) + 0.001), 3);
t = table(t);
t_total = [t_total; t];

%writetable(t, "z_timestep.csv")
%writetable(altitude, "z.csv")