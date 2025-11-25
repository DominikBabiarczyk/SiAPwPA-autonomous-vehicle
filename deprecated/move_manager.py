from vehicle_go import VehicleCommander

if __name__ == '__main__':
	commander = VehicleCommander()
	commander.go_vehicle(0.3, 0.2)  # Przykład: jedź z prędkością 0.3 m/s i skrętem 0.2 rad/s przez 0.5s
	commander.shutdown()
